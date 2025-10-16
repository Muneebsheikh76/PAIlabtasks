from flask import Flask, render_template, request, jsonify
import os, requests, base64
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Configuration ---
API_KEY = os.getenv("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q")
if not API_KEY:
    raise EnvironmentError("❌ Missing GEMINI_API_KEY in .env file")

MODEL = "models/gemini-2.5-pro"
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={API_KEY}"

# Store conversation (last 5 messages only)
chat_memory = []


# --- Utility Functions ---

def get_recent_context(limit=5):
    """Return recent messages for context."""
    return chat_memory[-limit:]

def encode_file_info(file_data):
    """Decode base64 and format file info."""
    try:
        b64_data = file_data["data"].split(",")[1]
        decoded = base64.b64decode(b64_data)
        mime_type = file_data.get("mime_type", "unknown")
        return f"[File attached: {mime_type}, {len(decoded)} bytes]"
    except Exception as e:
        raise ValueError(f"Invalid file data: {e}")

def send_to_gemini(context):
    """Send payload to Gemini API and return response text."""
    payload = {
        "contents": context,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()

    json_data = response.json()
    reply = (
        json_data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    return reply.strip()


# --- Routes ---

@app.route("/")
def index():
    """Main chat page."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Handle user questions and optional file upload."""
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    file_data = data.get("file")

    if not user_message and not file_data:
        return jsonify(error="Please provide a message or file."), 400

    # Build conversation context
    context = [{"role": msg["role"], "parts": msg["parts"]} for msg in get_recent_context()]

    # Handle file attachments differently
    if file_data and file_data.get("data"):
        try:
            file_info = encode_file_info(file_data)
            user_message = f"{user_message}\n{file_info}" if user_message else file_info
        except ValueError as e:
            return jsonify(error=str(e)), 400

    context.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        reply = send_to_gemini(context)
        if not reply:
            return jsonify(error="Empty response from Gemini."), 500

        # Save conversation
        chat_memory.extend([
            {"role": "user", "parts": [{"text": user_message}]},
            {"role": "model", "parts": [{"text": reply}]}
        ])

        return jsonify(response=reply)

    except requests.exceptions.RequestException as e:
        return jsonify(error=f"Gemini API request failed: {str(e)}"), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset chat history."""
    chat_memory.clear()
    return jsonify(message="Chat history cleared successfully.")


@app.route("/debug")
def debug():
    """Debug route to show internal info."""
    return jsonify({
        "api_key_loaded": bool(API_KEY),
        "model": MODEL,
        "stored_messages": len(chat_memory)
    })


# --- Run Server ---
if __name__ == "__main__":
    app.run(debug=True)
