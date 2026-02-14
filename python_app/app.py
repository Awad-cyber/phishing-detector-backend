import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from ml_model import predict, ModelNotLoadedError


app = Flask(__name__)

# Enable CORS for browser-based clients (Netlify, localhost, etc.)
# For a university project we keep this simple but controlled.
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def home():
    """
    Lightweight health-check endpoint for monitoring and Render.
    """
    return jsonify(
        {
            "status": "ok",
            "message": "Phishing detector API is running",
            "version": "2.0.0 (Flask)",
        }
    )


@app.route("/predict", methods=["POST"])
def predict_email():
    """
    Core prediction endpoint.

    Expects JSON:
        { "text": "<email body here>" }
    Returns JSON:
        { "success": true, "prediction": { ... } }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Email text is required for prediction.",
                }
            ),
            400,
        )

    try:
        result = predict(text)
        return jsonify({"success": True, "prediction": result}), 200
    except ModelNotLoadedError:
        # Clean, user-friendly message if the model files are missing
        return (
            jsonify(
                {
                    "success": False,
                    "error": "The detection model is not available on the server. "
                    "Please contact the system administrator.",
                }
            ),
            500,
        )
    except Exception:
        # Avoid leaking internal details in production-style code
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An unexpected error occurred while analysing the email.",
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Render typically sets PORT; default to 5000 for local development.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
