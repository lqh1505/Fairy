import os
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "GROQ_API_KEY_CUA_BAN_O_DAY")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

@app.get("/")
def root():
    return {"status": "Fairy Voice Assistant Server is running!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 Connected via WebSocket!")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            print(f"Nhận được {len(audio_bytes)} bytes audio từ ESP32")
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav"),
                "model": (None, "whisper-large-v3-turbo"),
                "language": (None, "vi"),
                "temperature": (None, "0")
            }
            response = requests.post(GROQ_URL, headers=headers, files=files)
            if response.status_code == 200:
                text_result = response.json().get("text", "")
                print(f"STT Result: {text_result}")
                await websocket.send_text(text_result)
            else:
                print(f"Lỗi từ Groq API: {response.text}")
                await websocket.send_text("[ERROR] Không nhận diện được giọng nói.")
    except WebSocketDisconnect:
        print("ESP32 Disconnected.")
