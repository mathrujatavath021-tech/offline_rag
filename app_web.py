import os
import json
import time
import datetime
import threading
import webbrowser
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, send_file
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Import RAG backend logic from rag.py
import rag

app = Flask(__name__, template_folder="templates")
FILES_DIR = os.path.join(rag.BASE_DIR, "FILES")
os.makedirs(FILES_DIR, exist_ok=True)

# Auto-load cached index from FILES on startup
if os.path.exists(FILES_DIR):
    loaded_count = rag.load_cached_index(FILES_DIR)
    print(f"[*] Initialized RAG Web Server: Loaded {loaded_count} cached document segments.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "ready": True,
        "chunks_count": len(rag.chunks),
        "model_loaded": rag.llm is not None,
        "current_session": rag.history_manager.current_session_id,
        "folder": "FILES"
    })


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    sessions = rag.history_manager.get_sessions()
    session_list = [{"id": sid, "title": title} for sid, title in sessions]
    return jsonify({
        "sessions": session_list,
        "current_id": rag.history_manager.current_session_id
    })


@app.route("/api/sessions/new", methods=["POST"])
def create_session():
    rag.history_manager.start_new_session()
    return jsonify({
        "session_id": rag.history_manager.current_session_id
    })


@app.route("/api/sessions/<path:session_id>", methods=["GET"])
def get_session_details(session_id):
    messages = rag.history_manager.get_session_messages(session_id)
    session_info = rag.history_manager.history_data.get(session_id, {})
    title = session_info.get("title", "Conversation")
    rag.history_manager.current_session_id = session_id
    return jsonify({
        "session_id": session_id,
        "title": title,
        "messages": messages
    })


@app.route("/api/sessions/<path:session_id>", methods=["DELETE"])
def delete_session(session_id):
    rag.history_manager.delete_sessions([session_id])
    if rag.history_manager.current_session_id == session_id:
        rag.history_manager.start_new_session()
    return jsonify({"success": True})


@app.route("/api/sessions/<path:session_id>/export", methods=["GET"])
def export_session(session_id):
    text_content = rag.history_manager.export_session_text(session_id)
    safe_name = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")) or "chat"
    return Response(
        text_content,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=chat_{safe_name}.txt"}
    )


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    selected_lang = data.get("lang", "English")
    session_id = data.get("session_id")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if session_id:
        rag.history_manager.current_session_id = session_id

    # Record user message
    rag.history_manager.save_message("user", query)

    def sse_generator():
        try:
            # Handle simple greetings without spinning LLM
            greetings = ["hi", "hello", "hlo", "hey", "hii", "hello ai"]
            if query.lower().strip() in greetings:
                greeting_resp = "Hello! I am your Offline AI."
                rag.history_manager.save_message("ai", greeting_resp)
                yield f"data: {json.dumps({'token': greeting_resp})}\n\n"
                yield f"data: {json.dumps({'done': True, 'answer': greeting_resp, 'session_id': rag.history_manager.current_session_id})}\n\n"
                return

            # Ensure LLM and Embedder are ready
            if rag.llm is None:
                rag.load_llm(lambda msg: None)

            if rag.embedder is None:
                rag.embedder = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)

            context_text = "No local documents found."
            matched_sources = []

            if len(rag.chunks) > 0 and len(rag.embeddings) > 0:
                q_vec = rag.embedder.encode([query])
                scores = cosine_similarity(q_vec, rag.embeddings)[0]
                top_indices = np.argsort(scores)[::-1][:3]
                matched_texts = []
                for idx in top_indices:
                    if scores[idx] > 0.25:
                        matched_texts.append(rag.chunks[idx]["text"])
                        matched_sources.append(rag.chunks[idx].get("source", "doc"))
                if matched_texts:
                    context_text = "\n\n".join(matched_texts)[:1500]

            current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
            wants_diagram = any(x in query.lower() for x in ["flowchart", "diagram"])

            if wants_diagram:
                prompt = f"""[INST] Draw a vertical ASCII flowchart for: "{query}".
Use [ ] for boxes and v for arrows. No markdown code blocks.
Context: {context_text} [/INST]"""
            else:
                prompt = f"""[INST] System: {current_time_str}.
Context: {context_text}
Question: {query}
If context is missing, answer using your own knowledge. [/INST]"""

            full_answer = ""
            if rag.llm:
                if selected_lang == "English":
                    stream = rag.llm(prompt, max_tokens=512, stop=["</s>"], stream=True)
                    for chunk in stream:
                        tok = chunk["choices"][0]["text"]
                        full_answer += tok
                        yield f"data: {json.dumps({'token': tok})}\n\n"
                else:
                    out = rag.llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
                    raw_ans = out["choices"][0]["text"].strip()
                    try:
                        lang_code = {"Hindi": "hi", "Telugu": "te", "Kannada": "kn"}.get(selected_lang, "en")
                        full_answer = GoogleTranslator(source="auto", target=lang_code).translate(raw_ans)
                    except Exception:
                        full_answer = raw_ans
                    yield f"data: {json.dumps({'token': full_answer})}\n\n"
            else:
                full_answer = "Error: LLM model could not be loaded. Please ensure the model exists in the models directory."
                yield f"data: {json.dumps({'token': full_answer})}\n\n"

            clean_answer = full_answer.strip()
            rag.history_manager.save_message("ai", clean_answer)
            yield f"data: {json.dumps({'done': True, 'answer': clean_answer, 'sources': list(set(matched_sources)), 'session_id': rag.history_manager.current_session_id})}\n\n"
        except (GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            print(f"Stream error: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception:
                pass

    return Response(sse_generator(), mimetype="text/event-stream")



@app.route("/api/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files part"}), 400

    uploaded = request.files.getlist("files")
    saved_any = False
    for file in uploaded:
        if file and file.filename:
            filename = secure_filename(file.filename)
            save_path = os.path.join(FILES_DIR, filename)
            file.save(save_path)
            saved_any = True

    if saved_any:
        success = rag.run_ingestion(FILES_DIR, lambda msg: None)
        return jsonify({
            "success": success,
            "chunks_count": len(rag.chunks)
        })
    return jsonify({"error": "No valid files received"}), 400


@app.route("/api/reindex", methods=["POST"])
def reindex_folder():
    success = rag.run_ingestion(FILES_DIR, lambda msg: None)
    return jsonify({
        "success": success,
        "chunks_count": len(rag.chunks)
    })


def open_browser(port):
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = 5000
    print("\n=======================================================")
    print("[*] Offline Multimodal RAG Assistant is running!")
    print(f"[*] Access in web browser at: http://127.0.0.1:{port}")
    print("=======================================================\n")
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)

