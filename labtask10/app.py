from flask import Flask, render_template, request, jsonify
from chatbot_logic import get_bot_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get", methods=["POST"])
def get_bot_reply():
    user_text = request.form.get("user_message", "")
    bot_reply = get_bot_response(user_text)
    return bot_reply
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Please send JSON with 'message' field"}), 400

    user_message = data["message"]
    bot_response = get_bot_response(user_message)

    return jsonify({
        "user": user_message,
        "bot": bot_response
    })

    app.run(host="0.0.0.0", port=5000, debug=True)
