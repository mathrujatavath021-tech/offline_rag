import os
import json
import time
import secrets
import hashlib
import datetime
from typing import Optional, Dict, Tuple, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTH_REGISTRY_PATH = os.path.join(DATA_DIR, "auth_registry.json")
USERS_DIR = os.path.join(DATA_DIR, "users")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

# In-memory sessions: session_token -> session_data
active_sessions: Dict[str, dict] = {}

# Rate limiting: ip_address -> [timestamp1, timestamp2, ...]
failed_login_attempts: Dict[str, list] = {}
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_LOCKOUT_SECONDS = 900  # 15 minutes
SESSION_EXPIRY_SECONDS = 86400 * 7  # 7 days


def _hash_code(code: str, salt: bytes) -> bytes:
    """Computes slow cryptographic hash using scrypt (standard library in Python 3.8+)."""
    normalized = code.strip().upper().encode("utf-8")
    return hashlib.scrypt(normalized, salt=salt, n=16384, r=8, p=1)


def generate_unique_code(prefix: str = "RAG") -> str:
    """Generates a cryptographically random, unambiguous access code: RAG-XXXX-XXXX-XXXX."""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # omit ambiguous 0, O, 1, I
    parts = ["".join(secrets.choice(chars) for _ in range(4)) for _ in range(3)]
    return f"{prefix}-{'-'.join(parts)}"


class AuthManager:
    def __init__(self):
        self.registry = self._load_registry()
        self._ensure_initial_admin()

    def _load_registry(self) -> dict:
        if os.path.exists(AUTH_REGISTRY_PATH):
            try:
                with open(AUTH_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Error loading auth registry: {e}")
                return {"users": {}}
        return {"users": {}}

    def _save_registry(self):
        temp_path = AUTH_REGISTRY_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=4)
        os.replace(temp_path, AUTH_REGISTRY_PATH)

    def _ensure_initial_admin(self):
        """Creates the default admin user if no users exist, and migrates existing data."""
        if not self.registry.get("users"):
            admin_code = "RAG-ADMIN-INIT-2026"
            salt = secrets.token_bytes(16)
            hashed = _hash_code(admin_code, salt)
            user_id = "user_default_admin"

            self.registry["users"][user_id] = {
                "user_id": user_id,
                "name": "Administrator",
                "role": "admin",
                "code_prefix": admin_code[:8],
                "salt_hex": salt.hex(),
                "hash_hex": hashed.hex(),
                "created_at": datetime.datetime.now().isoformat(),
                "is_revoked": False,
                "last_login": None
            }
            self._save_registry()
            self._initialize_user_storage(user_id)
            print(f"\n=======================================================")
            print(f"[*] INITIAL ADMIN ACCESS CODE: {admin_code}")
            print(f"[*] Use this code to log into the application.")
            print(f"=======================================================\n")

    def _initialize_user_storage(self, user_id: str):
        """Sets up private directories for a user and migrates existing data if default user."""
        user_dir = os.path.join(USERS_DIR, user_id)
        files_dir = os.path.join(user_dir, "files")
        cache_dir = os.path.join(user_dir, "cache")
        os.makedirs(files_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        user_history_file = os.path.join(user_dir, "chat_history.json")
        user_settings_file = os.path.join(user_dir, "settings.json")
        user_folders_file = os.path.join(user_dir, "monitored_folders.json")

        # Migrate existing root chat_history.json if present
        root_history = os.path.join(BASE_DIR, "chat_history.json")
        if not os.path.exists(user_history_file):
            if os.path.exists(root_history):
                try:
                    with open(root_history, "r", encoding="utf-8") as src, open(user_history_file, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                except Exception:
                    pass
            if not os.path.exists(user_history_file):
                with open(user_history_file, "w", encoding="utf-8") as f:
                    json.dump({}, f)

        # Initialize default monitored folder (FILES)
        if not os.path.exists(user_folders_file):
            default_files = os.path.join(BASE_DIR, "FILES")
            folders = [default_files] if os.path.exists(default_files) else []
            with open(user_folders_file, "w", encoding="utf-8") as f:
                json.dump({"folders": folders, "last_scan": None}, f, indent=2)

        # Default preferences
        if not os.path.exists(user_settings_file):
            with open(user_settings_file, "w", encoding="utf-8") as f:
                json.dump({
                    "theme": "dark",
                    "language": "English",
                    "tts_voice": "default",
                    "tts_speed": 1.0,
                    "voice_enabled": True,
                    "top_k": 3
                }, f, indent=2)

    def create_user(self, name: str, role: str = "user") -> Tuple[str, str]:
        """Generates a new access code and initializes isolated storage for the user."""
        code = generate_unique_code()
        salt = secrets.token_bytes(16)
        hashed = _hash_code(code, salt)
        user_id = f"user_{secrets.token_hex(6)}"

        self.registry["users"][user_id] = {
            "user_id": user_id,
            "name": name,
            "role": role,
            "code_prefix": code[:8],
            "salt_hex": salt.hex(),
            "hash_hex": hashed.hex(),
            "created_at": datetime.datetime.now().isoformat(),
            "is_revoked": False,
            "last_login": None
        }
        self._save_registry()
        self._initialize_user_storage(user_id)
        return code, user_id

    def verify_login(self, raw_code: str, client_ip: str = "127.0.0.1") -> Tuple[bool, Optional[str], Optional[dict], str]:
        """Validates an access code against Argon2/scrypt hashes with brute-force rate-limiting."""
        now = time.time()
        attempts = failed_login_attempts.get(client_ip, [])
        # Filter attempts within lockout window
        recent_attempts = [t for t in attempts if now - t < RATE_LIMIT_LOCKOUT_SECONDS]
        failed_login_attempts[client_ip] = recent_attempts

        if len(recent_attempts) >= RATE_LIMIT_MAX_ATTEMPTS:
            remaining_lockout = int(RATE_LIMIT_LOCKOUT_SECONDS - (now - recent_attempts[0]))
            return False, None, None, f"Too many failed attempts. Try again in {remaining_lockout} seconds."

        clean_code = raw_code.strip().upper()
        if not clean_code:
            return False, None, None, "Access code cannot be empty."

        for user_id, udata in self.registry.get("users", {}).items():
            if udata.get("is_revoked", False):
                continue

            try:
                salt = bytes.fromhex(udata["salt_hex"])
                stored_hash = bytes.fromhex(udata["hash_hex"])
                computed = _hash_code(clean_code, salt)

                if secrets.compare_digest(computed, stored_hash):
                    # Login successful -> reset failed attempts
                    failed_login_attempts[client_ip] = []
                    udata["last_login"] = datetime.datetime.now().isoformat()
                    self._save_registry()

                    # Create session
                    session_token = secrets.token_hex(32)
                    session_info = {
                        "user_id": user_id,
                        "name": udata.get("name", "User"),
                        "role": udata.get("role", "user"),
                        "created_at": now,
                        "expires_at": now + SESSION_EXPIRY_SECONDS
                    }
                    active_sessions[session_token] = session_info
                    return True, session_token, session_info, "Login successful."
            except Exception as e:
                print(f"[!] Auth comparison error: {e}")

        # Failed attempt
        failed_login_attempts.setdefault(client_ip, []).append(now)
        remaining = RATE_LIMIT_MAX_ATTEMPTS - len(failed_login_attempts[client_ip])
        return False, None, None, f"Invalid access code. ({remaining} attempts remaining before temporary lockout)"

    def validate_session(self, token: str) -> Optional[dict]:
        """Validates session token with expiration check."""
        if not token:
            return None
        sess = active_sessions.get(token)
        if not sess:
            return None
        if time.time() > sess["expires_at"]:
            del active_sessions[token]
            return None
        return sess

    def logout_session(self, token: str):
        if token in active_sessions:
            del active_sessions[token]

    def rotate_code(self, user_id: str) -> Optional[str]:
        """Rotates a user's access code, immediately invalidating the old one."""
        if user_id not in self.registry.get("users", {}):
            return None
        new_code = generate_unique_code()
        salt = secrets.token_bytes(16)
        hashed = _hash_code(new_code, salt)

        self.registry["users"][user_id]["salt_hex"] = salt.hex()
        self.registry["users"][user_id]["hash_hex"] = hashed.hex()
        self.registry["users"][user_id]["code_prefix"] = new_code[:8]
        self._save_registry()

        # Invalidate active sessions for this user
        to_del = [tok for tok, s in active_sessions.items() if s["user_id"] == user_id]
        for tok in to_del:
            del active_sessions[tok]

        return new_code

    def revoke_user(self, user_id: str) -> bool:
        """Revokes access for a user immediately."""
        if user_id in self.registry.get("users", {}):
            self.registry["users"][user_id]["is_revoked"] = True
            self._save_registry()
            to_del = [tok for tok, s in active_sessions.items() if s["user_id"] == user_id]
            for tok in to_del:
                del active_sessions[tok]
            return True
        return False

    def list_users(self) -> List[dict]:
        """Returns safe user listing without exposing cryptographic secrets."""
        users = []
        for uid, udata in self.registry.get("users", {}).items():
            users.append({
                "user_id": uid,
                "name": udata.get("name", "User"),
                "role": udata.get("role", "user"),
                "code_prefix": udata.get("code_prefix", "RAG-****") + "****",
                "created_at": udata.get("created_at"),
                "last_login": udata.get("last_login"),
                "is_revoked": udata.get("is_revoked", False)
            })
        return users

    def get_user_storage_path(self, user_id: str, subpath: str = "") -> str:
        """Returns isolated directory path for a user."""
        path = os.path.join(USERS_DIR, user_id)
        if subpath:
            path = os.path.join(path, subpath)
        os.makedirs(os.path.dirname(path) if "." in os.path.basename(path) else path, exist_ok=True)
        return path


auth_manager = AuthManager()
