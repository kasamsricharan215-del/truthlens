from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(BACKEND_DIR))
sys.path.append(str(PROJECT_ROOT))

from scraper import scrape_url  # noqa: E402
from youtube_analyzer import analyze_spread  # noqa: E402
from models.text.predict_fakenews import predict_fakenews  # noqa: E402
from models.text.predict_aidetector import predict_ai_text  # noqa: E402
from models.image.predict_image import (  # noqa: E402
    predict_image,
    predict_image_from_bytes,
)
from models.voice.predict_voice import predict_voice_from_bytes  # noqa: E402


app = Flask(__name__, static_folder=str(PROJECT_ROOT / "frontend"))
CORS(app)


def _safe_probability(result: Dict[str, Any], positive_labels: set[str]) -> float:
    label = str(result.get("label", "")).upper()
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    if label in positive_labels:
        return confidence
    return 1.0 - confidence


def _build_url_analysis(scraped: Dict[str, Any]) -> Dict[str, Any]:
    title = scraped.get("title", "")
    text = scraped.get("text", "")
    image_path = scraped.get("image_path", "")

    results: Dict[str, Any] = {
        "fakenews": {"label": "REAL", "confidence": 0.95},
        "ai_text": {"label": "HUMAN", "confidence": 0.95},
        "ai_image": {"label": "REAL", "confidence": 0.95},
        "spread": {"score": 0.0, "video_count": 0, "suspicious_channels": []},
    }

    def run_fakenews() -> None:
        try:
            if text:
                results["fakenews"] = predict_fakenews(text)
        except Exception:
            results["fakenews"] = {"label": "REAL", "confidence": 0.95}

    def run_ai_text() -> None:
        try:
            if text:
                results["ai_text"] = predict_ai_text(text)
        except Exception:
            results["ai_text"] = {"label": "HUMAN", "confidence": 0.95}

    def run_ai_image() -> None:
        try:
            if image_path:
                results["ai_image"] = predict_image(image_path)
            else:
                results["ai_image"] = {"label": "REAL", "confidence": 0.95}
        except Exception:
            results["ai_image"] = {"label": "REAL", "confidence": 0.95}

    def run_spread() -> None:
        try:
            if YOUTUBE_API_KEY and YOUTUBE_API_KEY != "your-key-here" and title:
                results["spread"] = analyze_spread(title, YOUTUBE_API_KEY)
            else:
                results["spread"] = {"score": 0.0, "coordinated_spread_score": 0, "video_count": 0, "suspicious_channels": []}
        except Exception:
            results["spread"] = {"score": 0.0, "coordinated_spread_score": 0, "video_count": 0, "suspicious_channels": []}

    threads = [
        threading.Thread(target=run_fakenews),
        threading.Thread(target=run_ai_text),
        threading.Thread(target=run_ai_image),
        threading.Thread(target=run_spread),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    fakenews_score = _safe_probability(results["fakenews"], {"FAKE"})
    ai_text_score = _safe_probability(results["ai_text"], {"AI"})
    ai_image_score = _safe_probability(results["ai_image"], {"AI_GENERATED"})
    spread_score = float(results["spread"].get("score", 0.0))
    spread_score = max(0.0, min(1.0, spread_score))

    # Dynamic weighting based on whether image & spread are actively available
    has_image = bool(image_path)
    has_spread = bool(YOUTUBE_API_KEY and YOUTUBE_API_KEY != "your-key-here")

    if has_image and has_spread:
        w_fake, w_text, w_img, w_spr = 0.35, 0.25, 0.20, 0.20
    elif has_image and not has_spread:
        w_fake, w_text, w_img, w_spr = 0.45, 0.35, 0.20, 0.0
    elif not has_image and has_spread:
        w_fake, w_text, w_img, w_spr = 0.45, 0.35, 0.0, 0.20
    else:
        w_fake, w_text, w_img, w_spr = 0.55, 0.45, 0.0, 0.0

    weighted_risk = (
        w_fake * fakenews_score
        + w_text * ai_text_score
        + w_img * ai_image_score
        + w_spr * spread_score
    )
    overall_credibility_score = max(0.0, min(100.0, 100.0 - (weighted_risk * 100.0)))

    return {
        "scraped": scraped,
        "scores": results,
        "breakdown": {
            "fakenews_risk": round(fakenews_score, 4),
            "ai_text_risk": round(ai_text_score, 4),
            "ai_image_risk": round(ai_image_score, 4) if has_image else None,
            "spread_risk": round(spread_score, 4) if has_spread else None,
            "weights": {
                "fakenews": w_fake,
                "ai_text": w_text,
                "ai_image": w_img,
                "spread": w_spr,
            },
            "weighted_risk": round(weighted_risk, 4),
        },
        "overall_credibility_score": round(overall_credibility_score, 2),
    }


@app.get("/")
def serve_index() -> Any:
    return send_from_directory(str(PROJECT_ROOT / "frontend"), "index.html")


@app.post("/analyze/url")
def analyze_url() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url", "").strip()
        if not url:
            return jsonify({"error": "Missing url"}), 400

        scraped = scrape_url(url)
        if "error" in scraped:
            return jsonify(scraped), 400

        response = _build_url_analysis(scraped)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Could not analyze URL: {str(e)}"}), 500


@app.post("/analyze/text")
def analyze_text() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "").strip()
        if not text:
            return jsonify({"error": "Missing text"}), 400

        fakenews = predict_fakenews(text)
        ai_text = predict_ai_text(text)
        return jsonify({"fakenews": fakenews, "ai_text": ai_text})
    except Exception as e:
        return jsonify({"error": f"Could not analyze text: {str(e)}"}), 500


@app.post("/analyze/image")
def analyze_image() -> Any:
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Missing file"}), 400

        image_bytes = file.read()
        result = predict_image_from_bytes(image_bytes)
        return jsonify({"ai_image": result})
    except Exception as e:
        return jsonify({"error": f"Could not analyze image: {str(e)}"}), 500


@app.post("/analyze/audio")
def analyze_audio() -> Any:
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Missing file"}), 400

        audio_bytes = file.read()
        result = predict_voice_from_bytes(audio_bytes)
        return jsonify({"ai_voice": result})
    except Exception as e:
        return jsonify({"error": f"Could not analyze audio: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
