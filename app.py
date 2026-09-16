"""
Flask backend for the domain-specific AI chatbot.

This file is generic and reusable across any domain — it never
hardcodes a specific college, company, profession, or domain. All
domain-specific behavior comes from chatbot_config.py, which is
regenerated per chatbot from a single title/purpose input.
"""

import os
import logging

from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

import chatbot_config as config

# =============================================================================
# CONSTANTS
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_HISTORY_MESSAGES = 20      # max stored turns kept from client history
MAX_MESSAGE_LENGTH = 4000      # max characters per message
ALLOWED_ROLES = {"user", "assistant"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

app = Flask(__name__)

# =============================================================================
# GEMINI CLIENT
# =============================================================================
_client = None


def get_client():
    """Lazily create and cache the GenAI client. Never expose the key."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# =============================================================================
# CONVERSATION HISTORY VALIDATION
# =============================================================================
def validate_and_clean_history(history):
    """
    Validate the client-supplied conversation history and return a
    cleaned list of {"role": ..., "content": ...} dicts, or raise
    ValueError with a safe message on invalid input.
    """
    if history is None:
        return []

    if not isinstance(history, list):
        raise ValueError("history must be a list.")

    if len(history) > 200:
        # Hard safety cap before trimming, to reject absurd payloads outright.
        raise ValueError("history is too long.")

    cleaned = []
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("Each history item must be an object.")

        role = item.get("role")
        content = item.get("content")

        if role not in ALLOWED_ROLES:
            raise ValueError("Each history item must have a valid role.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each history item must have text content.")

        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH]

        cleaned.append({"role": role, "content": content})

    # Keep only the most recent N messages so context stays bounded.
    return cleaned[-MAX_HISTORY_MESSAGES:]


def build_gemini_contents(history, message):
    """Turn validated history + current message into Gemini Content objects."""
    contents = []
    for item in history:
        gemini_role = "model" if item["role"] == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part(text=item["content"])],
            )
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=message)])
    )
    return contents


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        chatbot_title=config.CHATBOT_TITLE,
        chatbot_purpose=config.CHATBOT_PURPOSE,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON."}), 400

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "A non-empty 'message' string is required."}), 400

        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({"error": "Message is too long."}), 400

        try:
            history = validate_and_clean_history(body.get("history"))
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        contents = build_gemini_contents(history, message.strip())

        client = get_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=config.SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )

        reply_text = (response.text or "").strip()
        if not reply_text:
            reply_text = (
                "I'm sorry, I couldn't generate a response for that. "
                "Could you rephrase your question?"
            )

        return jsonify({"reply": reply_text}), 200

    except RuntimeError as re_err:
        # e.g. missing GEMINI_API_KEY
        logger.error("Configuration error: %s", re_err)
        return jsonify({"error": "The server is not configured correctly."}), 500
    except Exception:  # noqa: BLE001 - never leak internals to the client
        logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
