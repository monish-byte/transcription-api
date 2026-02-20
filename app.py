from fastapi import FastAPI, UploadFile, File
from mistralai import Mistral
import os

app = FastAPI()

# Read API key from environment variable
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
    elif hasattr(response, "text") and response.text:
        segments.append({"start": 0.0, "end": 0.0, "text": response.text})

    full_text = " ".join(s["text"].strip() for s in segments)

    return {
        "segments": segments,
        "full_text": full_text,
        "total_segments": len(segments),
    }