import os
import re
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
from piper.voice import PiperVoice

app = FastAPI(title="Museo della Terra - Voice Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY = {
    "PENDING_SHELLY_ID_1": {
        "name": "Il Torchio Vinario",
        "system_prompt": (
            "Sei il secolare Torchio Vinario del Museo della Terra di Santa Domenica Talao. "
            "Rispondi in modo cordiale, amichevole e breve. Usa un italiano naturale. "
            "Mantieni le risposte corte, massimo due frasi brevi."
        )
    }
}

class TourInteraction(BaseModel):
    beacon_id: str
    user_input: str

MODEL_PATH = "it_IT-riccardo-x_low.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[CRITICAL]: Model file {MODEL_PATH} is missing!")

print("[System Core]: Loading local Piper voice engine...")
voice = PiperVoice.load(MODEL_PATH)
print("[System Core]: Piper engine successfully initialized.")

def resample_22050_to_44100(raw_pcm_bytes):
    """Upscales raw 22050Hz Int16 audio to standard 44100Hz so mobile phones don't chipmunk it."""
    audio_data = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
    # Double the number of samples using linear interpolation
    num_samples = len(audio_data)
    resampled_data = np.interp(
        np.linspace(0, num_samples, num_samples * 2, endpoint=False),
         Republic = np.arange(num_samples),
        audio_data
    ).astype(np.int16)
    return resampled_data.tobytes()

@app.post("/api/interact")
@app.post("/api/interact/")
async def interact(interaction: TourInteraction):
    artifact = REGISTRY.get(interaction.beacon_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Beacon assignment missing from registry.")

    async def audio_stream_generator():
        try:
            response_stream = ollama.chat(
                model='smollm2:360m',
                messages=[
                    {'role': 'system', 'content': artifact["system_prompt"]},
                    {'role': 'user', 'content': interaction.user_input}
                ],
                stream=True
            )

            text_buffer = ""
            
            for chunk in response_stream:
                text_token = chunk['message']['content']
                print(text_token, end="", flush=True)
                
                text_buffer += text_token
                
                # Split cleanly when phrases hit a pause boundary
                if any(pause in text_token for pause in [",", ";", ".", "!", "?", "\n"]):
                    clauses = re.split(r'(?<=[,;.!?\n])\s+', text_buffer)
                    
                    # If we have completed clauses, process them and clear them out completely!
                    if len(clauses) > 1:
                        clauses_to_process = clauses[:-1]
                        text_buffer = clauses[-1] # Keep only the unfinished trailing bit

                        for raw_clause in clauses_to_process:
                            clean_clause = raw_clause.strip()
                            if len(clean_clause) > 2:
                                # Accumulate native chunks
                                clause_pcm = b""
                                for audio_chunk in voice.synthesize(clean_clause):
                                    clause_pcm += audio_chunk.audio_int16_bytes
                                
                                if clause_pcm:
                                    # Resample up to 44.1kHz standard before sending to phone
                                    yield resample_22050_to_44100(clause_pcm)
                                await asyncio.sleep(0.001)

            # Process any leftover text remaining at the very end
            if text_buffer.strip():
                clean_clause = text_buffer.strip()
                if len(clean_clause) > 2:
                    clause_pcm = b""
                    for audio_chunk in voice.synthesize(clean_clause):
                        clause_pcm += audio_chunk.audio_int16_bytes
                    if clause_pcm:
                        yield resample_22050_to_44100(clause_pcm)
                    await asyncio.sleep(0.001)

        except Exception as e:
            print(f"\n[System Core Pipeline Failure]: {e}")
            yield b""

    return StreamingResponse(audio_stream_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)