import os
import time
import json
import hashlib
import threading
import queue
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


import rag
import rag_engine

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt",
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".mp4", ".mkv", ".avi", ".mov",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
}

index_lock = threading.RLock()


def compute_file_hash(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a file efficiently in 64KB blocks."""
    if not os.path.isfile(filepath):
        return None
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[!] Hash error for {filepath}: {e}")
        return None


class IncrementalRAGHandler(FileSystemEventHandler):
    def __init__(self, watcher: "FolderWatcher"):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self.watcher.queue_event("created", event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self.watcher.queue_event("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self.watcher.queue_event("deleted", event.src_path)

    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory:
            self.watcher.queue_event("renamed", event.dest_path, old_path=event.src_path)


class FolderWatcher:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.registry_file = os.path.join(storage_dir, "file_registry.json")
        self.history_file = os.path.join(storage_dir, "file_activity.json")
        self.folders_file = os.path.join(storage_dir, "monitored_folders.json")
        self.cache_dir = os.path.join(storage_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.registry: Dict[str, dict] = self._load_json(self.registry_file, {})
        self.activity_log: List[dict] = self._load_json(self.history_file, [])
        self.monitored_folders: List[str] = self._load_json(self.folders_file, {}).get("folders", [])

        self.event_queue = queue.Queue()
        self.observer: Optional[Observer] = None
        self.is_paused = False
        self.is_running = True
        self.last_sync_time: Optional[str] = None

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self.worker_thread.start()

    def _load_json(self, path: str, default):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json(self, path: str, data):
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp, path)

    def log_activity(self, action: str, file_path: str, status: str, details: str = ""):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "filename": os.path.basename(file_path),
            "path": file_path,
            "status": status,
            "details": details
        }
        self.activity_log.insert(0, entry)
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[:100]
        self._save_json(self.history_file, self.activity_log)

    def queue_event(self, action: str, file_path: str, old_path: Optional[str] = None):
        if self.is_paused:
            return
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return
        # Ignore temp files or hidden files
        if os.path.basename(file_path).startswith((".", "~", "$")):
            return
        self.event_queue.put((action, file_path, old_path, time.time()))

    def _wait_file_stable(self, filepath: str, timeout: float = 3.0) -> bool:
        """Debounce: waits until file size stops changing."""
        t0 = time.time()
        last_size = -1
        while time.time() - t0 < timeout:
            if not os.path.exists(filepath):
                return False
            try:
                curr_size = os.path.getsize(filepath)
                if curr_size == last_size and curr_size > 0:
                    return True
                last_size = curr_size
            except Exception:
                pass
            time.sleep(0.5)
        return os.path.exists(filepath)

    def _process_queue_loop(self):
        while self.is_running:
            try:
                item = self.event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            action, file_path, old_path, q_time = item
            if self.is_paused:
                continue

            # Handle rename
            if action == "renamed" and old_path:
                self._handle_rename(old_path, file_path)
                continue

            # Handle delete
            if action == "deleted":
                self._handle_delete(file_path)
                continue

            # Handle created or modified
            if action in ("created", "modified"):
                # Wait for file write to complete
                if self._wait_file_stable(file_path):
                    self._handle_upsert(file_path, action)

    def _handle_rename(self, old_path: str, new_path: str):
        with index_lock:
            old_name = os.path.basename(old_path)
            new_name = os.path.basename(new_path)
            changed = False

            # Update registry
            if old_path in self.registry:
                info = self.registry.pop(old_path)
                info["path"] = new_path
                info["filename"] = new_name
                self.registry[new_path] = info
                self._save_json(self.registry_file, self.registry)

            # Update chunks in-memory and on disk
            for c in rag.chunks:
                if c.get("source") == old_name:
                    c["source"] = new_name
                    changed = True

            if changed:
                self._save_active_index()
                rag_engine.invalidate_cache()
                self.log_activity("renamed", new_path, "success", f"Renamed from {old_name}")

    def _handle_delete(self, file_path: str):
        with index_lock:
            fname = os.path.basename(file_path)
            if file_path in self.registry:
                del self.registry[file_path]
                self._save_json(self.registry_file, self.registry)

            # Remove chunks and embeddings for this file
            if rag.chunks and len(rag.chunks) > 0 and len(rag.embeddings) > 0:
                keep_indices = [i for i, c in enumerate(rag.chunks) if c.get("source") != fname]
                if len(keep_indices) < len(rag.chunks):
                    removed_count = len(rag.chunks) - len(keep_indices)
                    rag.chunks = [rag.chunks[i] for i in keep_indices]
                    rag.embeddings = rag.embeddings[keep_indices] if len(keep_indices) > 0 else np.empty((0, 384))
                    self._save_active_index()
                    rag_engine.invalidate_cache()
                    self.log_activity("deleted", file_path, "success", f"Removed {removed_count} segments")
                    return
            self.log_activity("deleted", file_path, "success", "Cleaned up record")

    def _handle_upsert(self, file_path: str, action: str):
        with index_lock:
            curr_hash = compute_file_hash(file_path)
            if not curr_hash:
                return

            # Check if unchanged
            prev_info = self.registry.get(file_path)
            if prev_info and prev_info.get("sha256") == curr_hash:
                return  # File unchanged, skip

            fname = os.path.basename(file_path)
            self.log_activity(action, file_path, "indexing", "Extracting text and embeddings...")

            # Extract text using existing multimodal parsers
            text = self._extract_text(file_path)
            if not text:
                self.log_activity(action, file_path, "skipped", "No text could be extracted")
                return

            words = text.split()
            new_file_chunks = []
            chunk_size = 500
            for j in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[j:j + chunk_size])
                if len(chunk_text) > 30:
                    new_file_chunks.append({"source": fname, "text": chunk_text})

            if not new_file_chunks:
                self.log_activity(action, file_path, "skipped", "Extracted text was under minimum threshold")
                return

            # Encode new chunks
            embedder = rag_engine.get_embedder()
            new_vecs = embedder.encode([c["text"] for c in new_file_chunks])

            # Remove existing chunks for this file if modifying
            if rag.chunks and len(rag.chunks) > 0 and len(rag.embeddings) > 0:
                keep_indices = [i for i, c in enumerate(rag.chunks) if c.get("source") != fname]
                base_chunks = [rag.chunks[i] for i in keep_indices]
                base_embeddings = rag.embeddings[keep_indices] if len(keep_indices) > 0 else np.empty((0, 384))
            else:
                base_chunks = []
                base_embeddings = np.empty((0, 384))

            # Merge
            if len(base_chunks) > 0:
                rag.chunks = base_chunks + new_file_chunks
                rag.embeddings = np.vstack([base_embeddings, new_vecs])
            else:
                rag.chunks = new_file_chunks
                rag.embeddings = new_vecs

            # Update registry
            self.registry[file_path] = {
                "path": file_path,
                "filename": fname,
                "sha256": curr_hash,
                "mtime": os.path.getmtime(file_path),
                "size": os.path.getsize(file_path),
                "chunk_count": len(new_file_chunks),
                "last_indexed": datetime.now().isoformat()
            }
            self._save_json(self.registry_file, self.registry)
            self._save_active_index()
            rag_engine.invalidate_cache()
            self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_activity(action, file_path, "success", f"Indexed {len(new_file_chunks)} segments")

    def _extract_text(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".pdf":
                import PyPDF2
                with open(filepath, "rb") as f:
                    r = PyPDF2.PdfReader(f)
                    return rag.clean_text(" ".join(pg.extract_text() or "" for pg in r.pages))
            elif ext == ".docx":
                from docx import Document
                d = Document(filepath)
                return rag.clean_text(" ".join(x.text for x in d.paragraphs))
            elif ext == ".txt":
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    return rag.clean_text(f.read())
            elif ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".avi", ".mov"):
                return rag.process_audio(filepath)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                return rag.process_image(filepath)
        except Exception as e:
            print(f"[!] Extraction error on {filepath}: {e}")
        return ""

    def _save_active_index(self):
        chunks_file = os.path.join(self.cache_dir, "chunks.json")
        embeddings_file = os.path.join(self.cache_dir, "embeddings.npy")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(rag.chunks, f)
        np.save(embeddings_file, rag.embeddings)

    def start_monitoring(self):
        """Initializes and starts watchdog observers for all registered folders."""
        if self.observer:
            self.stop_monitoring()

        self.observer = Observer()
        handler = IncrementalRAGHandler(self)
        any_folder = False

        for folder in self.monitored_folders:
            if os.path.isdir(folder):
                self.observer.schedule(handler, path=folder, recursive=False)
                any_folder = True

        if any_folder:
            self.observer.start()
            print(f"[*] Started file monitoring on folders: {self.monitored_folders}")

    def stop_monitoring(self):
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2.0)
            except Exception:
                pass
            self.observer = None

    def add_folder(self, folder_path: str) -> bool:
        norm = os.path.abspath(folder_path)
        if not os.path.isdir(norm):
            return False
        if norm not in self.monitored_folders:
            self.monitored_folders.append(norm)
            self._save_json(self.folders_file, {"folders": self.monitored_folders})
            self.start_monitoring()
            self.rescan_now()
        return True

    def remove_folder(self, folder_path: str) -> bool:
        norm = os.path.abspath(folder_path)
        if norm in self.monitored_folders:
            self.monitored_folders.remove(norm)
            self._save_json(self.folders_file, {"folders": self.monitored_folders})
            self.start_monitoring()
            return True
        return False

    def rescan_now(self):
        """Manually trigger incremental scan across all monitored folders."""
        for folder in self.monitored_folders:
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    full = os.path.join(folder, f)
                    if os.path.isfile(full):
                        self.queue_event("modified", full)
        self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Active per-user watcher instances: user_id -> FolderWatcher
user_watchers: Dict[str, FolderWatcher] = {}


def get_user_watcher(user_id: str) -> FolderWatcher:
    if user_id not in user_watchers:
        user_dir = os.path.join(rag.BASE_DIR, "data", "users", user_id)
        watcher = FolderWatcher(user_dir)
        watcher.start_monitoring()
        user_watchers[user_id] = watcher
    return user_watchers[user_id]
