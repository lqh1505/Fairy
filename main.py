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
                # FIX: verbose_json trả kèm no_speech_prob / avg_logprob theo từng
                # segment, giúp phát hiện và loại bỏ kết quả "hallucination"
                # (model bịa câu khi audio gần như im lặng/toàn tạp âm).
                "response_format": (None, "verbose_json"),
            }

            t_before_groq = time.time()
            try:
                response = await http_client.post(GROQ_URL, headers=headers, files=files)
                t_after_groq = time.time()
                print(f"[TIME] Gọi Groq mất: {t_after_groq - t_before_groq:.2f}s")

                if response.status_code == 200:
                    result_json = response.json()
                    text_result = result_json.get("text", "").strip()
                    segments = result_json.get("segments", [])

                    # FIX: kiểm tra độ tin cậy qua các segment để loại bỏ hallucination.
                    # no_speech_prob cao (gần 1) = model nghĩ đoạn đó không có giọng nói
                    # nhưng vẫn ráng in ra chữ -> chính là hallucination.
                    # avg_logprob rất âm (< -1.0) = model không tự tin vào kết quả.
                    is_hallucinated = False
                    if segments:
                        avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
                        avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
                        if avg_no_speech > 0.6 or avg_logprob < -1.0:
                            is_hallucinated = True
                        print(f"[DEBUG] no_speech_prob={avg_no_speech:.2f} avg_logprob={avg_logprob:.2f}")

                    if not text_result or is_hallucinated:
                        print(f"[INFO] Bỏ qua kết quả nghi ngờ hallucination: '{text_result}'")
                        await websocket.send_text("")
                    else:
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
