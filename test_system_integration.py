import os
import sys
import json
import io
import time
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app_web import app, auth_manager, rag_engine, voice_engine, file_watcher, rag

class SystemIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.admin_code = "RAG-ADMIN-INIT-2026"
        cls.admin_token = None

    def setUp(self):
        if not SystemIntegrationTest.admin_token:
            res = self.client.post("/api/auth/login", json={"code": self.admin_code})
            data = res.get_json()
            if data and data.get("token"):
                SystemIntegrationTest.admin_token = data["token"]

    def test_01_unauthenticated_access(self):
        """Unauthenticated requests should be rejected or report authenticated=False."""
        unauth_client = app.test_client()
        res = unauth_client.get("/api/auth/me")
        data = res.get_json()
        self.assertFalse(data.get("authenticated", False))

        # Accessing protected status endpoint without auth should return 401
        res = unauth_client.get("/api/status")
        self.assertEqual(res.status_code, 401)

    def test_02_admin_login(self):
        """Admin login with the initial unique code should succeed."""
        res = self.client.post("/api/auth/login", json={"code": self.admin_code})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data.get("success"))
        self.assertIn("token", data)
        SystemIntegrationTest.admin_token = data["token"]

        # Verify /api/auth/me returns admin profile
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/auth/me", headers=headers)
        data = res.get_json()
        self.assertTrue(data.get("authenticated"))
        self.assertEqual(data.get("role"), "admin")

    def test_03_rate_limiting(self):
        """5 failed login attempts should trigger a lockout."""
        test_ip = "192.168.1.99"
        for i in range(5):
            res = self.client.post(
                "/api/auth/login",
                json={"code": f"RAG-WRONG-CODE-{i}"},
                environ_base={"REMOTE_ADDR": test_ip}
            )
            self.assertEqual(res.status_code, 401)

        # 6th attempt must be locked out
        res = self.client.post(
            "/api/auth/login",
            json={"code": self.admin_code},
            environ_base={"REMOTE_ADDR": test_ip}
        )
        self.assertEqual(res.status_code, 401)
        msg = res.get_json().get("message", "").lower()
        self.assertTrue("too many failed attempts" in msg or "try again" in msg)

    def test_04_admin_user_management(self):
        """Admin should be able to create new users, get access code, rotate, and revoke."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 1. Create user
        res = self.client.post("/api/auth/users/create", json={"name": "Alice Test", "role": "user"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        user_id = data.get("user_id")
        code = data.get("access_code")
        self.assertTrue(code.startswith("RAG-"))

        # 2. Login with new user code
        res = self.client.post("/api/auth/login", json={"code": code})
        self.assertEqual(res.status_code, 200)
        user_token = res.get_json()["token"]

        # 3. New user cannot access admin endpoint
        res = self.client.get("/api/auth/users", headers={"Authorization": f"Bearer {user_token}"})
        self.assertEqual(res.status_code, 403)

        # 4. Admin list users
        res = self.client.get("/api/auth/users", headers=headers)
        self.assertEqual(res.status_code, 200)
        users = res.get_json()["users"]
        self.assertTrue(any(u["user_id"] == user_id for u in users))

        # 5. Rotate code
        res = self.client.post("/api/auth/users/rotate", json={"user_id": user_id}, headers=headers)
        self.assertEqual(res.status_code, 200)
        new_code = res.get_json()["new_code"]
        self.assertNotEqual(code, new_code)

        # 6. Revoke user
        res = self.client.post("/api/auth/users/revoke", json={"user_id": user_id}, headers=headers)
        self.assertEqual(res.status_code, 200)

        # Old code and new code should both fail now
        res = self.client.post("/api/auth/login", json={"code": new_code})
        self.assertEqual(res.status_code, 401)

    def test_05_voice_engine_endpoints(self):
        """TTS and STT endpoints should function offline without internet."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 1. Get TTS voices
        res = self.client.get("/api/voice/voices", headers=headers)
        self.assertEqual(res.status_code, 200)
        voices = res.get_json().get("voices", [])
        self.assertIsInstance(voices, list)

        # 2. Synthesize speech (TTS)
        res = self.client.post(
            "/api/voice/tts",
            json={"text": "System operational and ready.", "speed": 1.2},
            headers=headers
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "audio/wav")
        audio_bytes = res.data
        self.assertGreater(len(audio_bytes), 1000)

        # 3. Transcribe speech (STT) with Vosk using the synthesized WAV
        data = {
            "audio": (io.BytesIO(audio_bytes), "test_audio.wav")
        }
        res = self.client.post(
            "/api/voice/transcribe",
            data=data,
            content_type="multipart/form-data",
            headers=headers
        )
        self.assertEqual(res.status_code, 200)
        trans_data = res.get_json()
        self.assertTrue(trans_data.get("success"))
        transcript = trans_data.get("text", "").lower()
        self.assertTrue("operational" in transcript or "system" in transcript or "ready" in transcript)

    def test_06_file_monitoring_and_documents(self):
        """Monitor status, folder management, and document list should function."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 1. Get monitor status
        res = self.client.get("/api/monitor/status", headers=headers)
        self.assertEqual(res.status_code, 200)
        mon_data = res.get_json()
        self.assertIn("folders", mon_data)
        self.assertIn("is_paused", mon_data)

        # 2. Toggle monitor pause/resume
        res = self.client.post("/api/monitor/toggle", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["is_paused"])

        res = self.client.post("/api/monitor/toggle", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.get_json()["is_paused"])

        # 3. List documents
        res = self.client.get("/api/documents", headers=headers)
        self.assertEqual(res.status_code, 200)
        docs = res.get_json().get("documents", [])
        self.assertIsInstance(docs, list)

    def test_07_diagnostics(self):
        """Diagnostics endpoint should return accurate system resource information."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/diagnostics", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("cpu_percent", data)
        self.assertIn("rss_memory_mb", data)
        self.assertIn("system_cpu_cores", data)
        self.assertIn("system_threads", data)
        self.assertEqual(data["system_cpu_cores"], 4)
        self.assertEqual(data["system_threads"], 8)

    def test_08_chat_streaming_and_greetings(self):
        """Chat stream should stream greeting immediately without spinning LLM."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 1. Greeting query
        res = self.client.post(
            "/api/chat/stream",
            json={"query": "Hello", "lang": "English"},
            headers=headers
        )
        self.assertEqual(res.status_code, 200)
        raw_text = res.data.decode("utf-8")
        self.assertIn("Hello! I am your Offline AI.", raw_text)

    def test_09_rag_retrieval_and_stream(self):
        """End-to-end RAG query should retrieve context, stream tokens, cancel on token, and report diagnostics."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        req_id = f"test_req_{int(time.time()*1000)}"

        res = self.client.post(
            "/api/chat/stream",
            json={
                "query": "What are the phone space bar hacks?",
                "lang": "English",
                "request_id": req_id
            },
            headers=headers
        )
        self.assertEqual(res.status_code, 200)

        tokens = []
        done_event = None
        for chunk in res.response:
            line = chunk.decode("utf-8")
            for sub in line.split("\n\n"):
                if sub.startswith("data: "):
                    data = json.loads(sub[6:])
                    if "token" in data:
                        tokens.append(data["token"])
                        if len(tokens) == 5:
                            # Test live cancellation on 5th token
                            self.client.post(
                                "/api/chat/cancel",
                                json={"request_id": req_id},
                                headers=headers
                            )
                    if data.get("done"):
                        done_event = data

        self.assertIsNotNone(done_event, "Did not receive done event")
        self.assertGreater(len(tokens), 0, "No tokens streamed")
        self.assertIn("diagnostics", done_event)
        diag = done_event["diagnostics"]
        self.assertTrue("first_token_latency" in diag or "ttft_ms" in diag)
        self.assertTrue("tokens_per_sec" in diag or "tok_per_sec" in diag)
        self.assertIn("retrieval_ms", diag)
        print(f"\n[*] Verified RAG Stream & Live Cancellation: {len(tokens)} tokens, TTFT: {diag.get('first_token_latency', 0):.2f}s, Speed: {diag.get('tokens_per_sec', 0):.2f} tok/s, Retrieval: {diag['retrieval_ms']:.1f}ms")

if __name__ == "__main__":
    unittest.main()
