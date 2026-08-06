import os
import time
from datetime import datetime, timezone, timedelta
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # FIX: phục vụ favicon và các file tĩnh khác

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "GROQ_API_KEY_CUA_BAN_O_DAY")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# FIX: chọn nhà cung cấp LLM qua biến môi trường LLM_PROVIDER = "groq" | "gemini"
# Đặt biến này trên Render dashboard (Environment tab), mặc định là "groq"
# nếu không đặt gì.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

VN_TZ = timezone(timedelta(hours=7))  # FIX: múi giờ Việt Nam (UTC+7)

WEEKDAY_VI = ["Hai", "Ba", "Tư", "Năm", "Sáu", "Bảy", "Chủ Nhật"]

# FIX: trạng thái toàn cục để trang dashboard đọc và hiển thị real-time
dashboard_state = {
    "esp32_connected": False,
    "last_stt": "",
    "last_llm_reply": "",
    "last_updated": "",
}


def build_system_prompt() -> str:
    # FIX: tạo system prompt MỚI mỗi lần gọi, chèn ngày giờ thực tế theo giờ VN.
    # Trước đây prompt tĩnh khiến LLM không biết ngày hiện tại và tự bịa
    # (model chỉ có kiến thức tới lúc huấn luyện, không có đồng hồ thật).
    now = datetime.now(VN_TZ)
    weekday = WEEKDAY_VI[now.weekday()]
    date_str = f"Hôm nay là thứ {weekday}, ngày {now.strftime('%d/%m/%Y')}, giờ hiện tại là {now.strftime('%H:%M')}."
    return (
        f"Bạn là Fairy, một trợ lý ảo thân thiện, trả lời ngắn gọn, tự nhiên bằng tiếng Việt. "
        f"{date_str} "
        f"Trả lời trong 1-2 câu, không dùng markdown."
    )

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


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fairy Voice Assistant</title>
<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">
<link rel="apple-touch-icon" sizes="180x180" href="/static/favicon-180.png">
<style>
    body {
        background: #0f1117; color: #e6e6e6;
        font-family: -apple-system, Segoe UI, Roboto, sans-serif;
        display: flex; justify-content: center; padding: 40px 16px;
    }
    .card {
        background: #171a23; border-radius: 16px; padding: 28px 32px;
        max-width: 560px; width: 100%; box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    h1 { font-size: 22px; margin: 0 0 4px; }
    .sub { color: #8a8f9c; font-size: 13px; margin-bottom: 24px; }
    .status-row { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #555; transition: background 0.3s; }
    .dot.on { background: #3ddc84; box-shadow: 0 0 8px #3ddc84; }
    .block { background: #1e222d; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
    .label { color: #8a8f9c; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .value { font-size: 16px; line-height: 1.5; word-break: break-word; }
    .value.empty { color: #555; font-style: italic; }
    .footer { color: #555; font-size: 12px; margin-top: 20px; text-align: right; }
</style>
</head>
<body>
    <div class="card">
        <h1>🧚 Fairy Voice Assistant</h1>
        <div class="sub">Server đang chạy — dữ liệu tự cập nhật mỗi 2 giây</div>

        <div class="status-row">
            <div class="dot" id="dot"></div>
            <span id="connStatus">Đang kiểm tra kết nối ESP32...</span>
        </div>

        <div class="block">
            <div class="label">Bạn vừa nói</div>
            <div class="value" id="lastStt">—</div>
        </div>

        <div class="block">
            <div class="label">Fairy trả lời</div>
            <div class="value" id="lastReply">—</div>
        </div>

        <div class="footer" id="lastUpdated"></div>
    </div>

<script>
async function refresh() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('dot').className = 'dot' + (data.esp32_connected ? ' on' : '');
        document.getElementById('connStatus').innerText = data.esp32_connected ? 'ESP32 đang kết nối' : 'ESP32 chưa kết nối';
        setText('lastStt', data.last_stt);
        setText('lastReply', data.last_llm_reply);
        document.getElementById('lastUpdated').innerText = data.last_updated ? ('Cập nhật lần cuối: ' + data.last_updated) : '';
    } catch (e) {}
}
function setText(id, text) {
    const el = document.getElementById(id);
    if (text) { el.innerText = text; el.classList.remove('empty'); }
    else { el.innerText = 'Chưa có dữ liệu'; el.classList.add('empty'); }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"])  # FIX: hỗ trợ cả HEAD (UptimeRobot dùng để ping)
def root():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/status")
def status():
    return dashboard_state


async def call_groq_llm(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_LLM_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }
    resp = await http_client.post(GROQ_CHAT_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"[LLM-ERROR] Groq trả về {resp.status_code}: {resp.text}")
        return ""
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_gemini_llm(user_text: str) -> str:
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "system_instruction": {"parts": [{"text": build_system_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 150},
    }
    resp = await http_client.post(GEMINI_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"[LLM-ERROR] Gemini trả về {resp.status_code}: {resp.text}")
        return ""
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        print(f"[LLM-ERROR] Không parse được phản hồi Gemini: {data}")
        return ""


async def get_llm_reply(user_text: str) -> str:
    t0 = time.time()
    if LLM_PROVIDER == "gemini":
        reply = await call_gemini_llm(user_text)
    else:
        reply = await call_groq_llm(user_text)
    print(f"[TIME] Gọi LLM ({LLM_PROVIDER}) mất: {time.time() - t0:.2f}s")
    return reply


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 Connected via WebSocket!")
    dashboard_state["esp32_connected"] = True

    try:
        while True:
            # FIX: t_start giờ đặt SAU khi nhận xong audio, không phải trước.
            # Trước đây t_start đặt trước receive_bytes() nên "TỔNG thời gian"
            # vô tình cộng luôn cả khoảng ESP32 đang RẢNH chờ wake word tiếp theo
            # (có thể vài chục giây), khiến số liệu trông như bị "chậm" dù
            # thực ra STT+LLM chỉ mất chưa tới 1 giây.

            # Nhận dữ liệu audio dạng binary từ ESP32 gửi lên
            audio_bytes = await websocket.receive_bytes()
            t_start = time.time()
            print(f"[INFO] Nhận {len(audio_bytes)} bytes audio từ ESP32")

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
                        dashboard_state["last_stt"] = text_result
                        dashboard_state["last_updated"] = datetime.now(VN_TZ).strftime('%H:%M:%S %d/%m')

                        # FIX: gọi LLM để trả lời câu hỏi, chỉ in ra log để xem trước
                        # (chưa gửi ngược lại ESP32 - bước tiếp theo khi cần TTS/hiển thị).
                        llm_reply = await get_llm_reply(text_result)
                        if llm_reply:
                            print(f"[LLM Reply] {llm_reply}")
                            dashboard_state["last_llm_reply"] = llm_reply
                        else:
                            print("[LLM Reply] (không có phản hồi)")
                else:
                    print(f"Lỗi từ Groq API ({response.status_code}): {response.text}")
                    await websocket.send_text("[ERROR] Không nhận diện được giọng nói.")
            except httpx.TimeoutException:
                print("[ERROR] Groq API timeout!")
                await websocket.send_text("[ERROR] Server xử lý quá lâu, thử lại.")

            print(f"[TIME] TỔNG thời gian xử lý 1 lượt: {time.time() - t_start:.2f}s")

    except WebSocketDisconnect:
        print("ESP32 Disconnected.")
        dashboard_state["esp32_connected"] = False