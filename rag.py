import os
import json
import re
import threading
import sys
import math
import tempfile
import subprocess
import shutil
import time
import datetime
import queue
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from functools import partial
from PIL import Image, ImageTk

# --- MODERN GUI LIBRARY ---
try:
    import customtkinter as ctk
except Exception:
    ctk = None

# --- LIBRARY IMPORTS ---
try:
    import numpy as np
except Exception:
    np = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model, KaldiRecognizer = None, None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

VOSK_PATH = os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15")
LOCAL_EMBEDDING_DIR = os.path.join(MODELS_DIR, "embedding_model")
if os.path.exists(LOCAL_EMBEDDING_DIR):
    EMBEDDING_MODEL_NAME = LOCAL_EMBEDDING_DIR
else:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
SEPARATOR = "-" * 40


def find_tool(name):
    tool = shutil.which(name)
    if tool:
        return tool
    if name == "ffmpeg":
        candidates = [
            os.path.join(BASE_DIR, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe"),
            "ffmpeg/bin/ffmpeg.exe",
            "ffmpeg/ffmpeg.exe"
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
    elif name == "tesseract":
        candidates = [
            os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe"),
            "Tesseract-OCR/tesseract.exe"
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
    return name


FFMPEG_PATH = find_tool("ffmpeg")
TESSERACT_PATH = find_tool("tesseract")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --- GLOBAL OBJECTS ---
embedder = None
llm = None
chunks = []
embeddings = []
last_response_text = ""
current_engine = None
is_speaking = False
is_listening = False
audio_queue = queue.Queue()
is_busy = False

# --- SET THEME ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ==============================================================================
# PART 1: 3D ANIMATION ENGINE
# ==============================================================================

class AnimatedGlobe(ctk.CTkCanvas):
    def __init__(self, master, width, height, color="#00d4ff"):
        super().__init__(master, width=width, height=height, bg="#0f0c29", highlightthickness=0)
        self.width = width
        self.height = height
        self.color = color
        self.angle_x = 0
        self.angle_y = 0
        self.points = []
        self.scale = 250
        self.speed = 0.015

        self.generate_sphere()
        self.bind("<Configure>", self.on_resize)
        self.animate()

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height

    def generate_sphere(self):
        for i in range(0, 18):
            lat = math.pi * (i / 18.0 - 0.5)
            for j in range(0, 24):
                lon = 2 * math.pi * (j / 24.0)
                x = math.cos(lat) * math.cos(lon)
                y = math.cos(lat) * math.sin(lon)
                z = math.sin(lat)
                self.points.append([x, y, z])

    def animate(self):
        if not self.winfo_exists():
            return
        self.delete("all")
        rotation_z = [[math.cos(self.angle_x), -math.sin(self.angle_x), 0],
                      [math.sin(self.angle_x), math.cos(self.angle_x), 0], [0, 0, 1]]
        rotation_x = [[1, 0, 0], [0, math.cos(self.angle_y), -math.sin(self.angle_y)],
                      [0, math.sin(self.angle_y), math.cos(self.angle_y)]]

        for point in self.points:
            r = self.multiply_matrix(rotation_x, point)
            r = self.multiply_matrix(rotation_z, r)
            z_factor = 1 / (2 - r[2])
            x = r[0] * self.scale * z_factor + self.width / 2
            y = r[1] * self.scale * z_factor + self.height / 2
            size = 2 if r[2] > 0 else 1
            fill = self.color if r[2] > 0 else "#1a2f38"
            self.create_oval(x - size, y - size, x + size, y + size, fill=fill, outline="")

        self.angle_x += self.speed
        self.angle_y += self.speed * 0.6
        self.after(30, self.animate)

    def multiply_matrix(self, a, b):
        c = [0, 0, 0]
        c[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2]
        c[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2]
        c[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2]
        return c


# ==============================================================================
# PART 2: HISTORY MANAGER
# ==============================================================================

class HistoryManager:
    def __init__(self):
        self.filepath = HISTORY_FILE
        self.history_data = self.load_history()
        self.start_new_session()

    def start_new_session(self):
        self.current_session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.current_session_id not in self.history_data:
            self.history_data[self.current_session_id] = {
                "title": f"New Chat {datetime.datetime.now().strftime('%H:%M')}",
                "messages": []
            }

    def load_history(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_to_disk(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history_data, f, indent=4)

    def save_message(self, role, text):
        if self.current_session_id not in self.history_data:
            self.history_data[self.current_session_id] = {"title": "New Chat", "messages": []}
        session = self.history_data[self.current_session_id]
        if role == "user" and len(session["messages"]) == 0:
            session["title"] = text[:25] + "..." if len(text) > 25 else text
        session["messages"].append({"role": role, "text": text, "timestamp": str(datetime.datetime.now())})
        self.save_to_disk()

    def get_sessions(self):
        sessions = [(k, v.get("title", k)) for k, v in self.history_data.items()]
        return sorted(sessions, key=lambda x: x[0], reverse=True)

    def get_session_messages(self, session_id):
        return self.history_data.get(session_id, {}).get("messages", [])

    def delete_sessions(self, session_ids):
        for sid in session_ids:
            if sid in self.history_data:
                del self.history_data[sid]
        self.save_to_disk()

    def export_session_text(self, session_id):
        data = self.history_data.get(session_id)
        if not data:
            return ""
        output = f"Chat History: {data.get('title', 'Chat')}\nExported: {datetime.datetime.now()}\n{'=' * 40}\n\n"
        for msg in data.get('messages', []):
            sender = "User" if msg['role'] == "user" else "AI"
            output += f"[{sender}]: {msg['text']}\n{'-' * 20}\n"
        return output


history_manager = HistoryManager()


# ==============================================================================
# PART 3: INGESTION LOGIC
# ==============================================================================

def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()


def process_audio(p):
    if not os.path.exists(VOSK_PATH):
        return ""
    temp_wav = None
    try:
        read_path = p
        # If ffmpeg is available, ensure audio is resampled to 16kHz mono WAV
        if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
            fd, temp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = [FFMPEG_PATH, "-y", "-i", p, "-ar", "16000", "-ac", "1", temp_wav]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(temp_wav):
                read_path = temp_wav

        data, sr = sf.read(read_path, dtype="int16")
        if len(data.shape) > 1:
            data = data.mean(axis=1).astype("int16")

        model = Model(VOSK_PATH)
        rec = KaldiRecognizer(model, 16000)
        results = []
        for i in range(0, len(data), 4000):
            chunk = data[i:i + 4000].tobytes()
            if rec.AcceptWaveform(chunk):
                results.append(json.loads(rec.Result()).get("text", ""))
        results.append(json.loads(rec.FinalResult()).get("text", ""))
        return clean_text(" ".join(results))
    except Exception as e:
        print(f"Audio processing error on {p}: {e}")
        return ""
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def process_image(p):
    try:
        img = Image.open(p)
        text = pytesseract.image_to_string(img)
        return clean_text(text)
    except Exception as e:
        print(f"Image OCR error on {p}: {e}")
        return ""


def load_cached_index(target_folder):
    global chunks, embeddings
    local_cache_dir = os.path.join(target_folder, "cache")
    chunks_file = os.path.join(local_cache_dir, "chunks.json")
    embeddings_file = os.path.join(local_cache_dir, "embeddings.npy")
    if os.path.exists(chunks_file) and os.path.exists(embeddings_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                c = json.load(f)
            e = np.load(embeddings_file)
            if len(c) > 0 and len(e) == len(c):
                chunks = c
                embeddings = e
                return len(chunks)
        except Exception as ex:
            print(f"Error loading cache from {target_folder}: {ex}")
    return 0


def run_ingestion(target_folder, status_callback):
    global embedder, chunks, embeddings
    local_cache_dir = os.path.join(target_folder, "cache")
    chunks_file = os.path.join(local_cache_dir, "chunks.json")
    embeddings_file = os.path.join(local_cache_dir, "embeddings.npy")

    status_callback("Loading Embedder...")
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    status_callback(f"Scanning: {os.path.basename(target_folder)}")
    new_chunks = []
    corpus = []

    try:
        files = os.listdir(target_folder)
    except Exception as e:
        status_callback(f"Error: {e}")
        return False

    if not files:
        status_callback("Empty Folder")
        return False

    os.makedirs(local_cache_dir, exist_ok=True)

    for i, f in enumerate(files):
        p = os.path.join(target_folder, f)
        text = ""
        if not os.path.isfile(p):
            continue

        status_callback(f"Reading: {f}...")
        try:
            if f.lower().endswith(".pdf"):
                try:
                    with open(p, "rb") as f_pdf:
                        r = PyPDF2.PdfReader(f_pdf)
                        text = " ".join(pg.extract_text() or "" for pg in r.pages)
                except Exception as ex:
                    print(f"Error reading PDF {f}: {ex}")
                    text = ""
            elif f.lower().endswith(".docx"):
                try:
                    d = Document(p)
                    text = " ".join(x.text for x in d.paragraphs)
                except Exception as ex:
                    print(f"Error reading DOCX {f}: {ex}")
                    text = ""
            elif f.lower().endswith(".txt"):
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f_txt:
                        text = f_txt.read()
                except Exception as ex:
                    print(f"Error reading TXT {f}: {ex}")
                    text = ""
            elif f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".avi", ".mov")):
                text = process_audio(p)
            elif f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                text = process_image(p)

            text = clean_text(text)
            if not text:
                continue

            words = text.split()
            chunk_size = 500
            for j in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[j:j + chunk_size])
                if len(chunk_text) > 30:
                    new_chunks.append({"source": f, "text": chunk_text})
                    corpus.append(chunk_text)

        except Exception as e:
            print(f"Error processing {f}: {e}")

    if new_chunks:
        status_callback("Updating AI Memory...")
        vecs = embedder.encode(corpus)
        chunks = new_chunks
        embeddings = vecs
        with open(chunks_file, "w", encoding='utf-8') as f:
            json.dump(new_chunks, f)
        np.save(embeddings_file, vecs)
        status_callback(f"Success! Learned {len(new_chunks)} segments.")
        return True
    else:
        status_callback("No valid text found.")
        return False


def load_llm(callback):
    global llm
    if not os.path.exists(MODEL_PATH):
        callback(f"Error: Model missing at {MODEL_PATH}")
        return False

    if llm is None:
        try:
            callback("Loading Brain (GPU Mode)...")
            llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, n_gpu_layers=-1, verbose=False)
            callback("Brain Loaded (GPU Mode).")
            return True
        except Exception as e:
            try:
                callback(f"GPU Load note: {e}. Falling back to CPU...")
                llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
                callback("Brain Loaded (CPU Mode).")
                return True
            except Exception as e2:
                callback(f"LLM Crash: {e2}")
                return False
    return True


# ==============================================================================
# PART 4: GENERATION & LOGIC
# ==============================================================================

def speak_worker(text, on_finish):
    global current_engine, is_speaking
    try:
        import pythoncom
        pythoncom.CoInitialize()
        current_engine = pyttsx3.init()
        current_engine.setProperty('rate', 150)
        current_engine.say(text)
        current_engine.runAndWait()
    except Exception as e:
        print(f"Speech Error: {e}")
    finally:
        is_speaking = False
        if on_finish:
            on_finish()
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


def toggle_speech(button_widget):
    global is_speaking, current_engine, last_response_text, is_busy
    if is_busy:
        return

    if is_speaking:
        try:
            if current_engine:
                current_engine.stop()
        except Exception:
            pass
        is_speaking = False
        button_widget.configure(text="🔊 Speak Answer", fg_color="#4b5320", hover_color="#5f6b2e")
    else:
        if not last_response_text or len(last_response_text.strip()) == 0:
            return
        is_speaking = True
        button_widget.configure(text="⏹ Stop Speaking", fg_color="#8B0000", hover_color="#A52A2A")

        def reset_ui():
            button_widget.configure(text="🔊 Speak Answer", fg_color="#4b5320", hover_color="#5f6b2e")

        threading.Thread(target=speak_worker, args=(last_response_text, reset_ui), daemon=True).start()


def generate_diagram_code(context, query):
    if llm:
        prompt = f"""[INST] Create a vertical text-based flowchart for: "{query}".
        Use simple ASCII symbols (--> or v) and boxes [ ].
        Example:
        [ Start ]
           |
           v
        [ Input ]
        Context: {context[:800]} [/INST]"""
        output = llm(prompt, max_tokens=500, stop=["</s>"], echo=False)
        return output["choices"][0]["text"].strip()
    return "LLM Not Connected"


# ==============================================================================
# PART 5: MODERN GUI CLASS
# ==============================================================================

class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Offline Multimodal RAG Assistant")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Selected folder state
        self.selected_folder = None

        # --- 3D BACKGROUND ---
        self.bg_canvas = AnimatedGlobe(self, width=1200, height=800)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=15, fg_color="#1a1a2e", border_width=1,
                                    border_color="#333", bg_color="transparent")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        # IMPORTANT: Row 8 is History Frame (Expandable)
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.logo = ctk.CTkLabel(self.sidebar, text="RAG BRAIN",
                                 font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color="#00d4ff")
        self.logo.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Row 1: Select
        self.btn_select = ctk.CTkButton(self.sidebar, text="📂 Select Folder", command=self.select_folder, height=40,
                                        fg_color="#16213e", hover_color="#0f3460")
        self.btn_select.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # Row 2: Process
        self.btn_process = ctk.CTkButton(self.sidebar, text="⚙️ Process Docs", command=self.start_processing, height=40,
                                         fg_color="#E59937", hover_color="#D68826")
        self.btn_process.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        # Row 3: Label
        self.lbl_lang = ctk.CTkLabel(self.sidebar, text="Translation:", anchor="w", text_color="#a0a0a0")
        self.lbl_lang.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")

        # Row 4: Menu
        self.lang_var = ctk.StringVar(value="English")
        self.lang_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.lang_var,
                                           values=["English", "Hindi", "Telugu", "Kannada"], fg_color="#16213e",
                                           button_color="#0f3460")
        self.lang_menu.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        # Row 5: Speak
        self.btn_speak = ctk.CTkButton(self.sidebar, text="🔊 Speak Answer",
                                       command=lambda: toggle_speech(self.btn_speak), height=40, fg_color="#4b5320",
                                       hover_color="#5f6b2e")
        self.btn_speak.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

        # Row 6: New Chat
        self.btn_new_chat = ctk.CTkButton(self.sidebar, text="➕ New Chat", command=self.start_new_chat, height=40,
                                          fg_color="#2E8B57", hover_color="#3CB371")
        self.btn_new_chat.grid(row=6, column=0, padx=20, pady=(5, 5), sticky="ew")

        # Row 7: Manage History
        self.btn_manage = ctk.CTkButton(self.sidebar, text="⚙️ Manage History", command=self.open_manage_window,
                                        height=30, fg_color="#444", hover_color="#555")
        self.btn_manage.grid(row=7, column=0, padx=20, pady=5, sticky="ew")

        # Row 8: History List (Expands)
        self.history_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.history_frame.grid(row=8, column=0, padx=10, pady=5, sticky="nsew")

        # Row 9: Status
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready", text_color="#00e676", font=("Consolas", 12))
        self.lbl_status.grid(row=9, column=0, padx=20, pady=10, sticky="s")

        # --- MAIN CHAT ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#1a1a2e", border_width=1, border_color="#333",
                                       bg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 15), pady=15)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame, font=("Consolas", 11), wrap=tk.WORD,
            bg="#16213e", fg="white", bd=0, highlightthickness=0, padx=15, pady=15
        )
        self.chat_display.pack(expand=True, fill="both")
        self.chat_display.tag_config("user", foreground="#4da6ff", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("ai", foreground="#00e676", font=("Consolas", 11, "bold"))
        self.chat_display.configure(state="disabled")

        self.input_frame = ctk.CTkFrame(self.main_frame, height=60, corner_radius=20, fg_color="#1a1a2e",
                                        border_width=1, border_color="#444")
        self.input_frame.pack(fill="x", pady=(15, 0))

        self.entry_query = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question...", height=45,
                                        font=("Segoe UI", 14), border_width=0, fg_color="transparent")
        self.entry_query.pack(side="left", fill="x", expand=True, padx=15, pady=5)
        self.entry_query.bind("<KeyRelease>", self.toggle_send_mic_button)
        self.entry_query.bind("<Return>", lambda event: self.start_generation())

        self.btn_action = ctk.CTkButton(self.input_frame, text="🎤", width=50, height=40, command=self.handle_mic_click,
                                        corner_radius=15, fg_color="#333", hover_color="#444")
        self.btn_action.pack(side="right", padx=(5, 10))

        if not os.path.exists(MODEL_PATH):
            self.log(f"Warning: Model not found at {MODEL_PATH}.", "System")

        self.refresh_history_ui()

        # Auto-load default FILES directory if present
        default_folder = os.path.join(BASE_DIR, "FILES")
        if os.path.exists(default_folder):
            self.selected_folder = default_folder
            loaded = load_cached_index(default_folder)
            if loaded:
                self.lbl_status.configure(text=f"Ready: Loaded {loaded} cached segments")

    # --- CORE FUNCTIONS ---
    def start_generation(self):
        global last_response_text, is_busy
        query = self.entry_query.get().strip()
        if not query:
            return

        is_busy = True
        self.entry_query.delete(0, "end")
        self.btn_action.configure(text="⏳", state="disabled", fg_color="#555")

        self.log(query, "You")

        greetings = ["hi", "hello", "hlo", "hey", "hii", "hello ai"]
        if query.lower().strip() in greetings:
            greeting_resp = "Hello! I am your Offline AI."
            last_response_text = greeting_resp
            self.log(greeting_resp, "AI")
            self.finish_generation()
            return

        selected_lang = self.lang_var.get()
        wants_diagram = any(x in query.lower() for x in ["flowchart", "diagram"])

        def worker():
            global last_response_text, llm, embedder
            try:
                if llm is None:
                    load_llm(lambda msg: self.after(0, self.lbl_status.configure, {"text": msg}))

                self.after(0, self.lbl_status.configure, {"text": "Searching memory..."})
                context_text = "No local documents found."

                if len(chunks) > 0 and len(embeddings) > 0:
                    if embedder is None:
                        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
                    q_vec = embedder.encode([query])
                    scores = cosine_similarity(q_vec, embeddings)[0]
                    top_indices = np.argsort(scores)[::-1][:3]
                    matched_texts = []
                    for idx in top_indices:
                        if scores[idx] > 0.25:
                            matched_texts.append(chunks[idx]['text'])
                    if matched_texts:
                        context_text = "\n\n".join(matched_texts)[:1500]

                current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

                if wants_diagram:
                    prompt = f"""[INST] Draw a vertical ASCII flowchart for: "{query}".
                    Use [ ] for boxes and v for arrows. No markdown code blocks.
                    Context: {context_text} [/INST]"""
                else:
                    prompt = f"""[INST] System: {current_time_str}.
                    Context: {context_text}
                    Question: {query}
                    If context is missing, answer using your own knowledge. [/INST]"""

                self.after(0, self.lbl_status.configure, {"text": "Generating..."})

                if llm:
                    if selected_lang == "English":
                        self.after(0, lambda: self.chat_display.configure(state="normal"))
                        self.after(0, lambda: self.chat_display.insert("end", "🤖 AI: ", "ai"))
                        stream = llm(prompt, max_tokens=512, stop=["</s>"], stream=True)
                        full_answer = ""
                        for chunk in stream:
                            text_chunk = chunk['choices'][0]['text']
                            full_answer += text_chunk
                            self.after(0, self.stream_token, text_chunk)
                        self.after(0, lambda: self.chat_display.insert("end", "\n"))
                        answer = full_answer.strip()
                        history_manager.save_message("ai", answer)
                        self.after(0, self.refresh_history_ui)
                    else:
                        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
                        answer = output['choices'][0]['text'].strip()
                        try:
                            lang_code = {"Hindi": "hi", "Telugu": "te", "Kannada": "kn"}.get(selected_lang, "en")
                            answer = GoogleTranslator(source='auto', target=lang_code).translate(answer)
                        except Exception:
                            pass
                        self.after(0, self.log, answer, "AI")
                else:
                    answer = "LLM not connected. Please check model path."
                    self.after(0, self.log, answer, "AI")

                last_response_text = answer

            except Exception as e:
                self.after(0, self.lbl_status.configure, {"text": f"Error: {str(e)[:40]}..."})
            finally:
                self.after(0, self.finish_generation)

        threading.Thread(target=worker, daemon=True).start()

    # --- BOILERPLATE UI METHODS ---
    def finish_generation(self):
        global is_busy
        is_busy = False
        self.btn_action.configure(state="normal")
        self.lbl_status.configure(text="Ready")
        self.toggle_send_mic_button()

    def toggle_send_mic_button(self, event=None):
        global is_busy
        if is_busy:
            return
        text = self.entry_query.get().strip()
        if len(text) > 0:
            self.btn_action.configure(text="➤", fg_color="#00d4ff", hover_color="#00a3cc",
                                      command=self.start_generation)
        else:
            self.btn_action.configure(text="🎤", fg_color="#333", hover_color="#444", command=self.handle_mic_click)

    def handle_mic_click(self):
        global is_listening, is_busy
        if is_busy:
            return
        if is_listening:
            is_listening = False
            self.btn_action.configure(text="🎤", fg_color="#333")
            self.lbl_status.configure(text="Voice Stopped")
        else:
            is_listening = True
            self.btn_action.configure(text="🛑", fg_color="#ff4444", hover_color="#cc0000")
            self.lbl_status.configure(text="Listening... Speak Now")
            threading.Thread(target=self.voice_listener, daemon=True).start()

    def voice_listener(self):
        global is_listening
        if not os.path.exists(VOSK_PATH):
            self.after(0, self.lbl_status.configure, {"text": "Vosk model missing"})
            is_listening = False
            self.btn_action.configure(text="🎤", fg_color="#333")
            return

        model = Model(VOSK_PATH)
        rec = KaldiRecognizer(model, 16000)

        def callback(indata, frames, time_info, status):
            audio_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
                while is_listening:
                    if not audio_queue.empty():
                        data = audio_queue.get()
                        if rec.AcceptWaveform(data):
                            res = json.loads(rec.Result())
                            if res.get("text"):
                                self.after(0, lambda t=res["text"]: self.entry_query.insert("end", t + " "))
                                self.after(0, self.toggle_send_mic_button)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            print(f"Microphone error: {e}")
        finally:
            try:
                final_res = json.loads(rec.FinalResult())
                if final_res.get("text"):
                    self.after(0, lambda t=final_res["text"]: self.entry_query.insert("end", t + " "))
                    self.after(0, self.toggle_send_mic_button)
            except Exception:
                pass
            is_listening = False
            self.after(0, self.start_generation)

    def open_manage_window(self):
        win = ctk.CTkToplevel(self)
        win.title("Manage History")
        win.geometry("500x500")
        win.attributes('-topmost', True)
        bg_frame = ctk.CTkFrame(win, fg_color="#1a1a2e")
        bg_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(bg_frame, text="Select Chats to Download or Delete", font=("Segoe UI", 16, "bold"),
                     text_color="white").pack(pady=10)
        scroll_frame = ctk.CTkScrollableFrame(bg_frame, width=450, height=350, fg_color="#16213e")
        scroll_frame.pack(pady=10)
        sessions = history_manager.get_sessions()
        self.check_vars = {}
        for session_id, title in sessions:
            row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            var = tk.BooleanVar()
            self.check_vars[session_id] = var
            chk = ctk.CTkCheckBox(row, text="", variable=var, width=20, border_color="#aaa")
            chk.pack(side="left", padx=5)
            lbl = ctk.CTkLabel(row, text=title[:45], anchor="w", text_color="#ddd")
            lbl.pack(side="left", padx=5, fill="x", expand=True)
            btn_dl = ctk.CTkButton(row, text="💾", width=30, fg_color="#444", hover_color="#555",
                                   command=partial(self.download_chat, session_id))
            btn_dl.pack(side="right", padx=5)
        btn_del = ctk.CTkButton(bg_frame, text="🗑️ Delete Selected", fg_color="#ff4444", hover_color="#cc0000",
                                command=lambda: self.delete_selected_chats(win))
        btn_del.pack(pady=10)

    def download_chat(self, session_id):
        text_content = history_manager.export_session_text(session_id)
        if not text_content:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            messagebox.showinfo("Success", "Chat history downloaded successfully.")

    def delete_selected_chats(self, window):
        to_delete = [sid for sid, var in self.check_vars.items() if var.get()]
        if not to_delete:
            return
        if messagebox.askyesno("Confirm", f"Delete {len(to_delete)} chats?"):
            history_manager.delete_sessions(to_delete)
            window.destroy()
            self.refresh_history_ui()
            if history_manager.current_session_id in to_delete:
                self.start_new_chat()

    def start_new_chat(self):
        history_manager.start_new_session()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.refresh_history_ui()

    def refresh_history_ui(self):
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        sessions = history_manager.get_sessions()
        for session_id, title in sessions:
            btn = ctk.CTkButton(self.history_frame, text=title, fg_color="transparent", border_width=1,
                                border_color="#333", anchor="w", height=30,
                                command=partial(self.load_session_to_chat, session_id))
            btn.pack(fill="x", pady=2)

    def load_session_to_chat(self, session_id):
        history_manager.current_session_id = session_id
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        messages = history_manager.get_session_messages(session_id)
        for msg in messages:
            role_tag = "user" if msg['role'] == "user" else "ai"
            sender_name = "👤 You: " if msg['role'] == "user" else "🤖 AI: "
            self.chat_display.insert("end", f"\n{sender_name}", role_tag)
            self.chat_display.insert("end", f"{msg['text']}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def log(self, message, sender="System"):
        if sender == "System":
            self.lbl_status.configure(text=message[:40] + "..." if len(message) > 40 else message)
            return
        self.chat_display.configure(state="normal")
        if sender == "You":
            self.chat_display.insert("end", f"\n\n👤 You: {message}\n", "user")
            history_manager.save_message("user", message)
            self.after(0, self.refresh_history_ui)
        elif sender == "AI":
            self.chat_display.insert("end", f"🤖 AI: {message}\n", "ai")
            history_manager.save_message("ai", message)
            self.after(0, self.refresh_history_ui)
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def stream_token(self, token):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", token, "ai")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=BASE_DIR)
        if folder:
            self.selected_folder = folder
            loaded = load_cached_index(folder)
            if loaded:
                self.log(f"Selected: {os.path.basename(folder)} (Loaded {loaded} cached segments)", "System")
            else:
                self.log(f"Selected: {os.path.basename(folder)}", "System")

    def start_processing(self):
        if not self.selected_folder:
            messagebox.showwarning("Error", "Please select a folder first.")
            return
        self.btn_process.configure(state="disabled", text="Processing...")

        def worker():
            success_ingest = run_ingestion(self.selected_folder, lambda msg: self.after(0, self.log, msg, "System"))
            if success_ingest:
                self.after(0, self.log, "Ingestion complete. Connecting to LLM...", "System")
                load_llm(lambda msg: self.after(0, self.log, msg, "System"))
            else:
                self.after(0, self.log, "Ingestion failed.", "System")
            self.after(0, lambda: self.btn_process.configure(state="normal", text="⚙️ Process Docs"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    if "--web" in sys.argv or "-w" in sys.argv:
        from app_web import app as web_app, open_browser
        port = 5000
        print("\n[*] Starting Offline Multimodal RAG Assistant (Web Mode)...")
        print(f"[*] Access in web browser at: http://127.0.0.1:{port}\n")
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
        web_app.run(host="127.0.0.1", port=port, debug=False)
    else:
        if ctk is None:
            print("customtkinter or Tkinter is not available in this environment. Run in web mode: python app_web.py")
            sys.exit(1)
        app = RAGApp()
        app.mainloop()