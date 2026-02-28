from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

try:
    from src.sms_fraud.inference import load_best, predict_proba
    from src.sms_fraud.chatbot import get_response, get_response_local
    numpy_available = True
except ImportError:
    numpy_available = False
    load_best = None
    predict_proba = None
    get_response = None
    get_response_local = None


app = Flask(__name__, template_folder="templates", static_folder="static")


def _load_meta() -> dict:
    meta_path = Path("models/best/meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


loaded = None
meta: dict = {}
threshold: float = 0.5

if numpy_available:
    try:
        print("Loading model...")
        loaded = load_best("models/best")
        meta = _load_meta()
        threshold = float(meta.get("threshold", 0.5))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        loaded = None
else:
    print("Warning: NumPy not available. Model loading skipped.")


def get_loaded_model():
    return loaded


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    loaded = get_loaded_model()
    if loaded is None:
        return jsonify({"error": "No trained model. Run: python train.py --model_type sklearn"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400

    probs = predict_proba(loaded, [text])
    proba = float(probs[0])
    is_fraud = proba >= threshold
    return jsonify(
        {
            "fraud_probability": proba,
            "is_fraud": bool(is_fraud),
            "threshold": threshold,
        }
    )

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if get_response is None:
        print("Chatbot not available")
        return jsonify({"error": "Chatbot not available"}), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    backend = (data.get("backend") or "local").lower()
    # Use env var for OpenAI, or allow override from request for flexibility
    api_key = data.get("api_key") or os.getenv("OPENAI_API_KEY")
    
    if not question:
        return jsonify({"error": "Empty question"}), 400

    if backend == "openai" and not api_key:
        return jsonify({"error": "OpenAI API key required"}), 400

    print(f"Chat request backend={backend} question={question[:50]}...")
    try:
        from src.sms_fraud.chatbot import ChatMessage

        history_data = data.get("history", []) or []
        history = []
        for m in history_data:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content and isinstance(content, str):
                history.append(ChatMessage(role=role, content=content.strip()))

        model = "gpt-4o-mini" if backend == "openai" else (data.get("local_model") or "google/flan-t5-small")
        response = get_response(question, history, backend=backend, api_key=api_key or None, model=model)
        response = str(response).strip() if response else "No response generated. Try rephrasing your question."
        print(f"Chat response length: {len(response)}")
        return jsonify({"response": response})
    except Exception as e:
        err_msg = str(e) if e else "Unknown error"
        print(f"Chat error: {err_msg}")
        return jsonify({"error": f"Chat failed: {err_msg}"}), 500


@app.route("/api/batch_predict", methods=["POST"])
def api_batch_predict():
    loaded = get_loaded_model()
    if loaded is None:
        return jsonify({"error": "No trained model. Run: python train.py --model_type sklearn"}), 500

    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide a list of texts"}), 400

    probs = predict_proba(loaded, texts)
    results = []
    for i, proba in enumerate(probs):
        proba = float(proba)
        is_fraud = proba >= threshold
        results.append({
            "text": texts[i],
            "fraud_probability": proba,
            "is_fraud": bool(is_fraud),
        })
    return jsonify({"results": results, "threshold": threshold})


if __name__ == "__main__":
    try:
        from waitress import serve
        print("Starting server on http://localhost:5000")
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        print("Error: waitress module not found. Install it with: pip install waitress")
        print("Or run with: python web_app.py")

