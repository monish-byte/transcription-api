from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from mistralai import Mistral
import os
import uuid

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TRANSCRIPT_DIR = "transcripts"

# Ensure transcript directory exists
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Transcription API is running"}


# ---------------------------
# TRANSCRIBE ENDPOINT
# ---------------------------
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)

        audio_bytes = await file.read()

        response = client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file={
                "content": audio_bytes,
                "file_name": file.filename,
            },
            timestamp_granularities=["segment"],
        )

        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append(seg.text)
        elif hasattr(response, "text") and response.text:
            segments.append(response.text)

        full_text = " ".join(s.strip() for s in segments)

        # Generate unique filename
        transcript_id = str(uuid.uuid4())
        filename = f"{transcript_id}.txt"
        file_path = os.path.join(TRANSCRIPT_DIR, filename)

        # Save transcript locally
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        public_url = f"/transcripts/{filename}"

        return {
            "message": "Transcription saved successfully",
            "transcript_url": public_url,
            "full_text": full_text
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# SERVE INDIVIDUAL TRANSCRIPT AS HTML
# ---------------------------
@app.get("/transcripts/{filename}", response_class=HTMLResponse)
def get_transcript(filename: str):
    file_path = os.path.join(TRANSCRIPT_DIR, filename)

    if not os.path.exists(file_path):
        return HTMLResponse(
            content="<h1>Transcript not found</h1>",
            status_code=404
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    html_content = f"""
    <html>
        <head>
            <title>Transcript</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Transcript</h1>
            <div style="white-space: pre-wrap; font-family: Arial; line-height: 1.6;">
                {content}
            </div>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


# ---------------------------
# LIST ALL TRANSCRIPTS (FOR KORE.AI CRAWLER)
# ---------------------------
@app.get("/all-transcripts", response_class=HTMLResponse)
def list_transcripts():
    files = os.listdir(TRANSCRIPT_DIR)

    if not files:
        return HTMLResponse(
            content="<h1>No transcripts available yet.</h1>"
        )

    links = ""
    for file in files:
        links += f'<li><a href="/transcripts/{file}">{file}</a></li>'

    html_content = f"""
    <html>
        <head>
            <title>All Transcripts</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>All Transcripts</h1>
            <ul>
                {links}
            </ul>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)