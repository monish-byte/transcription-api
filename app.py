from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from mistralai import Mistral
import os
import uuid

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TRANSCRIPT_DIR = "transcripts"

# Create transcripts folder if not exists
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Transcription API is running"}


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):

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
    file_path = os.path.join(TRANSCRIPT_DIR, f"{transcript_id}.txt")

    # Save transcript
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    public_url = f"/transcripts/{transcript_id}.txt"

    return {
        "message": "Transcription saved successfully",
        "transcript_url": public_url,
        "full_text": full_text
    }


@app.get("/transcripts/{filename}")
def get_transcript(filename: str):
    file_path = os.path.join(TRANSCRIPT_DIR, filename)
    return FileResponse(file_path)