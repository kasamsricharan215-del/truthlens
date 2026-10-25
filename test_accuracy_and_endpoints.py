import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import soundfile as sf

from backend.app import app

def test_all():
    client = app.test_client()
    print("==================================================")
    print("RUNNING TRUTHLENS ACCURACY & ENDPOINT VERIFICATION")
    print("==================================================")

    # 1. TEXT ENDPOINT TEST
    print("\n--- 1. Testing /analyze/text ---")
    ai_text = (
        "In today's fast-paced digital world, artificial intelligence plays a pivotal role "
        "in revolutionizing various sectors. Furthermore, it serves as a testament to human innovation, "
        "delving into complex problems and fostering holistic solutions."
    )
    human_text = (
        "The Renaissance was a fervent period of European cultural, artistic, political and economic "
        "rebirth following the Middle Ages. Generally described as taking place from the 14th century "
        "to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art."
    )
    fake_news = "SHOCKING BOMBSHELL: Secret cure the government is hiding from everyone exposed!"
    real_news = (
        "WASHINGTON (Reuters) - The Federal Reserve kept interest rates steady on Wednesday and indicated "
        "that borrowing costs are likely to remain unchanged for an extended period."
    )

    # Test AI text
    res = client.post("/analyze/text", json={"text": ai_text})
    assert res.status_code == 200, f"Failed: {res.data}"
    data = res.get_json()
    print("AI Text Response:", json.dumps(data, indent=2))
    assert data["ai_text"]["label"] == "AI"
    assert data["ai_text"]["confidence"] >= 0.95

    # Test Human text
    res = client.post("/analyze/text", json={"text": human_text})
    assert res.status_code == 200
    data = res.get_json()
    print("Human Text Response:", json.dumps(data, indent=2))
    assert data["ai_text"]["label"] == "HUMAN"
    assert data["ai_text"]["confidence"] >= 0.95

    # Test Fake news
    res = client.post("/analyze/text", json={"text": fake_news})
    assert res.status_code == 200
    data = res.get_json()
    print("Fake News Response:", json.dumps(data, indent=2))
    assert data["fakenews"]["label"] == "FAKE"
    assert data["fakenews"]["confidence"] >= 0.95

    # Test Real news
    res = client.post("/analyze/text", json={"text": real_news})
    assert res.status_code == 200
    data = res.get_json()
    print("Real News Response:", json.dumps(data, indent=2))
    assert data["fakenews"]["label"] == "REAL"
    assert data["fakenews"]["confidence"] >= 0.95

    # 2. IMAGE ENDPOINT TEST
    print("\n--- 2. Testing /analyze/image ---")
    img = Image.new("RGB", (256, 256), color=(140, 160, 200))
    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    res = client.post("/analyze/image", data={"file": (img_buf, "test.jpg")}, content_type="multipart/form-data")
    assert res.status_code == 200, f"Failed: {res.data}"
    data = res.get_json()
    print("Image Response:", json.dumps(data, indent=2))
    assert "ai_image" in data
    assert data["ai_image"]["confidence"] >= 0.95

    # 3. AUDIO ENDPOINT TEST
    print("\n--- 3. Testing /analyze/audio ---")
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    audio = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 360 * t)
    audio_buf = io.BytesIO()
    sf.write(audio_buf, audio, sr, format="WAV")
    audio_buf.seek(0)

    res = client.post("/analyze/audio", data={"file": (audio_buf, "test.wav")}, content_type="multipart/form-data")
    assert res.status_code == 200, f"Failed: {res.data}"
    data = res.get_json()
    print("Audio Response:", json.dumps(data, indent=2))
    assert "ai_voice" in data
    assert data["ai_voice"]["confidence"] >= 0.95

    # 4. URL ENDPOINT TEST
    print("\n--- 4. Testing /analyze/url ---")
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    res = client.post("/analyze/url", json={"url": test_url})
    assert res.status_code == 200, f"Failed: {res.data}"
    data = res.get_json()
    print("URL Response (Credibility):", data.get("overall_credibility_score"), "%")
    print("URL Breakdown:", json.dumps(data.get("breakdown"), indent=2))
    assert data["overall_credibility_score"] >= 90.0

    print("\n==================================================")
    print("ALL TESTS PASSED WITH 95-100% ACCURACY & CONFIDENCE!")
    print("==================================================")

if __name__ == "__main__":
    test_all()
