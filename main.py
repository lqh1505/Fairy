import os
import time
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "GROQ_API_KEY_CUA_BAN_O_DAY")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# FIX: tạo 1 AsyncClient dùng chung cho cả app, có connection pool + keep-alive
# tới Groq. Việc này tránh phải bắt tay TCP/TLS mới (thường tốn 200-800ms)
# cho MỖI câu nói, vì trước đó "requests.post" tạo kết nối mới mỗi lần gọi.
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=15.0)


@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()


@app.get("/")
def root():
    return {"status": "Fairy Voice Assistant Server is running!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 Connected via WebSocket!")

    try:
        while True:
            t_start = time.time()

            # Nhận dữ liệu audio dạng binary từ ESP32 gửi lên
            audio_bytes = await websocket.receive_bytes()
            t_received = time.time()
            print(f"[TIME] Nhận {len(audio_bytes)} bytes mất: {t_received - t_start:.2f}s")

            # Đẩy sang Groq API để làm STT (dùng client dùng chung, giữ kết nối)
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav"),
                "model": (None, "whisper-large-v3-turbo"),
                "language": (None, "vi"),
                "temperature": (None, "0"),
            }

            t_before_groq = time.time()
            try:
                response = await http_client.post(GROQ_URL, headers=headers, files=files)
                t_after_groq = time.time()
                print(f"[TIME] Gọi Groq mất: {t_after_groq - t_before_groq:.2f}s")

                if response.status_code == 200:
                    text_result = response.json().get("text", "")
                    print(f"STT Result: {text_result}")
                    await websocket.send_text(text_result)
                else:
                    print(f"Lỗi từ Groq API ({response.status_code}): {response.text}")
                    await websocket.send_text("[ERROR] Không nhận diện được giọng nói.")
            except httpx.TimeoutException:
                print("[ERROR] Groq API timeout!")
                await websocket.send_text("[ERROR] Server xử lý quá lâu, thử lại.")

            print(f"[TIME] TỔNG thời gian xử lý 1 lượt: {time.time() - t_start:.2f}s")

    except WebSocketDisconnect:
        print("ESP32 Disconnected.")
