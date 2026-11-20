import os
import time
import json
import psutil
import datetime
import threading
import webbrowser
from functools import wraps
from typing import Optional
from flask import Flask, render_template, request, Response, jsonify, send_file, make_response
from werkzeug.utils import secure_filename

import rag
import rag_engine
from auth_manager import auth_manager
import voice_engine
import file_watcher

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# Global start time
SERVER_START_TIME = time.time()

# Pre-load shared RAG model on startup if available
print("[*] Pre-warming RAG Engine...")
try:
    rag_engine.get_embedder()
    rag_engine.get_llm()
except Exception as e:
    print(f"[*] Pre-warming note: {e}")

# Initialize default monitored folder for default admin
try:
    default_admin_watcher = file_watcher.get_user_watcher("user_default_admin")
except Exception as e:
    print(f"[*] File watcher init note: {e}")


# ==============================================================================
# AUTHENTICATION DECORATOR & HELPERS
# ==============================================================================

def get_current_user() -> Optional[dict]:
    token = None
    if request.headers.get("Authorization"):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("session_token")
    if token:
        return auth_manager.validate_session(token)
    return None


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Unauthorized", "auth_required": True}), 401
        request.user = user
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Unauthorized", "auth_required": True}), 401
        if user.get("role") != "admin":
            return jsonify({"error": "Forbidden: Admin access required"}), 403
        request.user = user
        return f(*args, **kwargs)
    return decorated


def get_user_history_path(user_id: str) -> str:
    user_dir = auth_manager.get_user_storage_path(user_id)
    history_file = os.path.join(user_dir, "chat_history.json")
    if not os.path.exists(history_file):
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return history_file


def load_user_history(user_id: str) -> dict:
    hpath = get_user_history_path(user_id)
    try:
        with open(hpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_history(user_id: str, data: dict):
    hpath = get_user_history_path(user_id)
    temp = hpath + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp, hpath)


# ==============================================================================
# FRONTEND ENTRY POINT
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ==============================================================================
# AUTHENTICATION APIS
# ==============================================================================

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    code = data.get("code", "").strip()
    client_ip = request.remote_addr or "127.0.0.1"

    ok, token, user_info, msg = auth_manager.verify_login(code, client_ip=client_ip)
    if not ok or not token:
        return jsonify({"success": False, "message": msg}), 401

    resp = make_response(jsonify({
        "success": True,
        "token": token,
        "user": user_info,
        "message": msg
    }))
    resp.set_cookie("session_token", token, max_age=86400 * 7, httponly=False, samesite="Lax")
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = request.cookies.get("session_token") or (request.headers.get("Authorization") or "")[7:]
    if token:
        auth_manager.logout_session(token)
    resp = make_response(jsonify({"success": True}))
    resp.delete_cookie("session_token")
    return resp


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "user_id": user["user_id"],
        "name": user.get("name", "User"),
        "role": user.get("role", "user")
    })


@app.route("/api/auth/users", methods=["GET"])
@require_admin
def list_users():
    return jsonify({"users": auth_manager.list_users()})


@app.route("/api/auth/users/create", methods=["POST"])
@require_admin
def create_user():
    data = request.json or {}
    name = data.get("name", "New User").strip()
    role = data.get("role", "user")
    code, user_id = auth_manager.create_user(name=name, role=role)
    return jsonify({
        "success": True,
        "user_id": user_id,
        "access_code": code,
        "message": "Generated new access code. Display this code to the user now (it cannot be viewed again in plaintext)."
    })


@app.route("/api/auth/users/rotate", methods=["POST"])
@require_admin
def rotate_user_code():
    data = request.json or {}
    user_id = data.get("user_id")
    new_code = auth_manager.rotate_code(user_id)
    if new_code:
        return jsonify({"success": True, "new_code": new_code})
    return jsonify({"error": "User not found"}), 404


@app.route("/api/auth/users/revoke", methods=["POST"])
@require_admin
def revoke_user_code():
    data = request.json or {}
    user_id = data.get("user_id")
    ok = auth_manager.revoke_user(user_id)
    return jsonify({"success": ok})


# ==============================================================================
# RAG, CHAT & SESSIONS APIS
# ==============================================================================

@app.route("/api/status", methods=["GET"])
@require_auth
def get_system_status():
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    return jsonify({
        "ready": True,
        "chunks_count": len(rag.chunks),
        "model_loaded": rag_engine._llm_instance is not None,
        "user": request.user,
        "monitored_folders_count": len(watcher.monitored_folders),
        "last_sync": watcher.last_sync_time,
        "uptime_seconds": int(time.time() - SERVER_START_TIME)
    })


@app.route("/api/sessions", methods=["GET"])
@require_auth
def get_user_sessions():
    user_id = request.user["user_id"]
    history = load_user_history(user_id)
    sessions = []
    for sid, sdata in history.items():
        sessions.append({
            "id": sid,
            "title": sdata.get("title", "Chat"),
            "count": len(sdata.get("messages", []))
        })
    sessions.sort(key=lambda x: x["id"], reverse=True)
    return jsonify({"sessions": sessions})


@app.route("/api/sessions/new", methods=["POST"])
@require_auth
def create_new_session():
    user_id = request.user["user_id"]
    history = load_user_history(user_id)
    sid = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history[sid] = {
        "title": f"New Chat {datetime.datetime.now().strftime('%H:%M')}",
        "messages": []
    }
    save_user_history(user_id, history)
    return jsonify({"session_id": sid})


@app.route("/api/sessions/<path:session_id>", methods=["GET"])
@require_auth
def get_session_content(session_id):
    user_id = request.user["user_id"]
    history = load_user_history(user_id)
    sdata = history.get(session_id, {})
    return jsonify({
        "session_id": session_id,
        "title": sdata.get("title", "Conversation"),
        "messages": sdata.get("messages", [])
    })


@app.route("/api/sessions/<path:session_id>", methods=["DELETE"])
@require_auth
def delete_user_session(session_id):
    user_id = request.user["user_id"]
    history = load_user_history(user_id)
    if session_id in history:
        del history[session_id]
        save_user_history(user_id, history)
    return jsonify({"success": True})


@app.route("/api/sessions/<path:session_id>/export", methods=["GET"])
@require_auth
def export_user_session(session_id):
    user_id = request.user["user_id"]
    history = load_user_history(user_id)
    sdata = history.get(session_id)
    if not sdata:
        return "Session not found", 404

    lines = [f"Chat History: {sdata.get('title', 'Chat')}", f"Exported: {datetime.datetime.now()}", "=" * 40, ""]
    for msg in sdata.get("messages", []):
        sender = "User" if msg["role"] == "user" else "AI"
        lines.append(f"[{sender}]: {msg['text']}")
        lines.append("-" * 20)

    safe_name = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")) or "chat"
    return Response(
        "\n".join(lines),
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=chat_{safe_name}.txt"}
    )


@app.route("/api/chat/cancel", methods=["POST"])
@require_auth
def cancel_generation():
    data = request.json or {}
    req_id = data.get("request_id")
    if req_id:
        rag_engine.cancel_request(req_id)
        return jsonify({"success": True, "cancelled": req_id})
    return jsonify({"error": "Missing request_id"}), 400


@app.route("/api/chat/stream", methods=["POST"])
@require_auth
def chat_stream():
    user_id = request.user["user_id"]
    data = request.json or {}
    query = (data.get("query") or "").strip()
    selected_lang = data.get("lang", "English")
    session_id = data.get("session_id")
    request_id = data.get("request_id") or f"req_{int(time.time()*1000)}"

    if not query:
        return jsonify({"error": "Empty query"}), 400

    history = load_user_history(user_id)
    if not session_id or session_id not in history:
        session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history[session_id] = {
            "title": query[:25] + "..." if len(query) > 25 else query,
            "messages": []
        }

    session_entry = history[session_id]
    if len(session_entry["messages"]) == 0:
        session_entry["title"] = query[:25] + "..." if len(query) > 25 else query

    session_entry["messages"].append({
        "role": "user",
        "text": query,
        "timestamp": str(datetime.datetime.now())
    })
    save_user_history(user_id, history)

    def sse_generator():
        # Handle simple greetings without spinning LLM
        greetings = ["hi", "hello", "hlo", "hey", "hii", "hello ai"]
        if query.lower().strip() in greetings:
            greeting_resp = "Hello! I am your Offline AI."
            session_entry["messages"].append({
                "role": "ai",
                "text": greeting_resp,
                "timestamp": str(datetime.datetime.now())
            })
            save_user_history(user_id, history)
            yield f"data: {json.dumps({'token': greeting_resp})}\n\n"
            yield f"data: {json.dumps({'done': True, 'answer': greeting_resp, 'sources': [], 'session_id': session_id})}\n\n"
            return

        # Retrieve relevant context
        with file_watcher.index_lock:
            context_text, sources, ret_lat = rag_engine.RAGEngine.retrieve_context(
                query=query,
                chunks=rag.chunks,
                embeddings=rag.embeddings,
                top_k=3
            )

        full_ans = ""
        try:
            for item in rag_engine.RAGEngine.stream_answer(
                query=query,
                context_text=context_text,
                matched_sources=sources,
                selected_lang=selected_lang,
                request_id=request_id
            ):
                if "token" in item:
                    full_ans += item["token"]
                    yield f"data: {json.dumps(item)}\n\n"
                elif item.get("done"):
                    clean_ans = item["answer"].strip()
                    session_entry["messages"].append({
                        "role": "ai",
                        "text": clean_ans,
                        "timestamp": str(datetime.datetime.now())
                    })
                    save_user_history(user_id, history)
                    item["session_id"] = session_id
                    diag = item.setdefault("diagnostics", {})
                    diag["retrieval_ms"] = ret_lat
                    diag["first_token_latency"] = round(diag.get("ttft_ms", 0) / 1000.0, 2)
                    diag["total_generation_time"] = round(diag.get("total_ms", 0) / 1000.0, 2)
                    diag["tokens_per_sec"] = diag.get("tok_per_sec", 0.0)
                    yield f"data: {json.dumps(item)}\n\n"

        except (GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(sse_generator(), mimetype="text/event-stream")


# ==============================================================================
# OFFLINE VOICE APIS (STT & TTS)
# ==============================================================================

@app.route("/api/voice/voices", methods=["GET"])
@require_auth
def get_tts_voices():
    voices = voice_engine.get_available_voices()
    return jsonify({"voices": voices})


@app.route("/api/voice/tts", methods=["POST"])
@require_auth
def text_to_speech():
    data = request.json or {}
    text = data.get("text", "")
    voice_id = data.get("voice_id")
    speed = float(data.get("speed", 1.0))

    ok, wav_path, msg = voice_engine.synthesize_speech(text, voice_id=voice_id, speed=speed)
    if not ok or not wav_path:
        return jsonify({"error": msg}), 400

    def cleanup_after_send():
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

    threading.Timer(10.0, cleanup_after_send).start()
    return send_file(wav_path, mimetype="audio/wav")


@app.route("/api/voice/transcribe", methods=["POST"])
@require_auth
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    filename = secure_filename(audio_file.filename)
    ext = os.path.splitext(filename)[1].lower() or ".wav"

    ok, text, msg = voice_engine.transcribe_audio_file(audio_bytes, suffix=ext)
    if ok:
        return jsonify({"success": True, "text": text, "message": msg})
    return jsonify({"success": False, "error": msg}), 400


# ==============================================================================
# FILE MONITORING & DOCUMENT MANAGEMENT APIS
# ==============================================================================

@app.route("/api/monitor/status", methods=["GET"])
@require_auth
def get_monitor_status():
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    return jsonify({
        "folders": watcher.monitored_folders,
        "is_paused": watcher.is_paused,
        "last_sync": watcher.last_sync_time,
        "activity_log": watcher.activity_log[:30],
        "total_files_tracked": len(watcher.registry)
    })


@app.route("/api/monitor/folders", methods=["POST"])
@require_auth
def add_monitored_folder():
    user_id = request.user["user_id"]
    data = request.json or {}
    folder = data.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Invalid folder path"}), 400

    watcher = file_watcher.get_user_watcher(user_id)
    ok = watcher.add_folder(folder)
    return jsonify({"success": ok, "folders": watcher.monitored_folders})


@app.route("/api/monitor/folders", methods=["DELETE"])
@require_auth
def remove_monitored_folder():
    user_id = request.user["user_id"]
    data = request.json or {}
    folder = data.get("folder", "").strip()
    watcher = file_watcher.get_user_watcher(user_id)
    ok = watcher.remove_folder(folder)
    return jsonify({"success": ok, "folders": watcher.monitored_folders})


@app.route("/api/monitor/rescan", methods=["POST"])
@require_auth
def rescan_monitored_folders():
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    watcher.rescan_now()
    return jsonify({"success": True, "message": "Triggered folder rescan."})


@app.route("/api/monitor/toggle", methods=["POST"])
@require_auth
def toggle_monitoring():
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    watcher.is_paused = not watcher.is_paused
    return jsonify({"success": True, "is_paused": watcher.is_paused})


@app.route("/api/documents", methods=["GET"])
@require_auth
def list_documents():
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    docs = []
    for path, info in watcher.registry.items():
        docs.append({
            "filename": info.get("filename", os.path.basename(path)),
            "path": path,
            "size_kb": round(info.get("size", 0) / 1024, 1),
            "chunk_count": info.get("chunk_count", 0),
            "last_indexed": info.get("last_indexed", "")
        })
    return jsonify({"documents": docs, "total_chunks": len(rag.chunks)})


@app.route("/api/upload", methods=["POST"])
@require_auth
def upload_and_index():
    user_id = request.user["user_id"]
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    user_files_dir = auth_manager.get_user_storage_path(user_id, "files")
    uploaded = request.files.getlist("files")
    saved_paths = []

    for f in uploaded:
        if f and f.filename:
            fname = secure_filename(f.filename)
            target = os.path.join(user_files_dir, fname)
            f.save(target)
            saved_paths.append(target)

    # Trigger incremental index for each saved file
    watcher = file_watcher.get_user_watcher(user_id)
    for p in saved_paths:
        watcher._handle_upsert(p, "uploaded")

    return jsonify({
        "success": True,
        "files_count": len(saved_paths),
        "total_chunks": len(rag.chunks)
    })


@app.route("/api/documents/<filename>", methods=["DELETE"])
@require_auth
def delete_document(filename):
    user_id = request.user["user_id"]
    watcher = file_watcher.get_user_watcher(user_id)
    target_path = None
    for p, info in watcher.registry.items():
        if info.get("filename") == filename or os.path.basename(p) == filename:
            target_path = p
            break

    if target_path:
        watcher._handle_delete(target_path)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass
        return jsonify({"success": True, "message": f"Deleted {filename}."})
    return jsonify({"error": "Document not found"}), 404


# ==============================================================================
# PERFORMANCE & SYSTEM DIAGNOSTICS API
# ==============================================================================

@app.route("/api/diagnostics", methods=["GET"])
@require_auth
def get_diagnostics():
    proc = psutil.Process()
    mem_info = proc.memory_info()
    cpu_pct = psutil.cpu_percent(interval=None)

    return jsonify({
        "cpu_percent": cpu_pct,
        "rss_memory_mb": round(mem_info.rss / 1024 / 1024, 1),
        "vms_memory_mb": round(mem_info.vms / 1024 / 1024, 1),
        "system_cpu_cores": psutil.cpu_count(logical=False),
        "system_threads": psutil.cpu_count(logical=True),
        "model_path": os.path.basename(rag.MODEL_PATH),
        "embedding_model": os.path.basename(rag.EMBEDDING_MODEL_NAME),
        "query_cache_size": len(rag_engine._query_embed_cache),
        "generation_cache_size": len(rag_engine._generation_cache),
        "active_cancellations": len(rag_engine.active_cancellations),
        "chunks_indexed": len(rag.chunks),
        "uptime_formatted": str(datetime.timedelta(seconds=int(time.time() - SERVER_START_TIME)))
    })


# ==============================================================================
# SERVER LAUNCHER
# ==============================================================================

def open_browser(port: int):
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = 5000
    print("\n=======================================================")
    print("[*] Offline Multimodal RAG Assistant (Enhanced Edition)")
    print(f"[*] Access in web browser at: http://127.0.0.1:{port}")
    print("=======================================================\n")
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
