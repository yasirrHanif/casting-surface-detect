"""Usage: APP_API_KEY='your-secret' python client_example.py path/to/image.jpg"""
import mimetypes
import os
import sys
import requests

if len(sys.argv) != 2:
    raise SystemExit("Usage: python client_example.py path/to/image.jpg")

key = os.getenv("APP_API_KEY")
if not key:
    raise SystemExit("Set APP_API_KEY first")

url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
content_type, _ = mimetypes.guess_type(sys.argv[1])
if content_type not in {"image/jpeg", "image/png", "image/webp"}:
    raise SystemExit("Supported extensions: .jpg, .jpeg, .png, .webp")
with open(sys.argv[1], "rb") as f:
    response = requests.post(
        url,
        headers={"X-API-Key": key},
        files={"file": (os.path.basename(sys.argv[1]), f, content_type)},
        timeout=60,
    )
response.raise_for_status()
print(response.json())
