import os
import re
import asyncio
import struct
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
    """Upscales raw 22050Hz Int16 audio to standard 44100Hz."""
    audio_data = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
    num_samples = len(audio_data)
    resampled_data = np.interp(
        np.linspace(0, num_samples, num_samples * 2, endpoint=False),
        np.arange(num_samples),
        audio_data
    ).astype(np.int16)
    return resampled_data

def create_wav_chunk(pcm_data_16bit, sample_rate=44100):
    """Wraps raw 16-bit PCM numpy arrays inside a standard browser-readable WAV container."""
    raw_bytes = pcm_data_16bit.tobytes()
    num_bytes = len(raw_bytes)
    
    # 44-byte RIFF/WAV standard header structure
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        num_bytes + 36,  # Total file chunk size minus 8 bytes
        b'WAVE',
        b'fmt ',
        16,              # Subchunk1Size (16 for PCM)
        1,               # AudioFormat (1 for uncompressed PCM)
        1,               # NumChannels (1 for Mono)
        sample_rate,     # SampleRate
        sample_rate * 2, # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
        2,               # BlockAlign (NumChannels * BitsPerSample/8)
        16,              # BitsPerSample
        b'data',
        num_bytes        # Subchunk2Size
    )
    return header + raw_bytes

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
                
                if any(pause in text_token for pause in [",", ";", ".", "!", "?", "\n"]):
                    clauses = re.split(r'(?<=[,;.!?\n])\s+', text_buffer)
                    
                    if len(clauses) > 1:
                        clauses_to_process = clauses[:-1]
                        text_buffer = clauses[-1]

                        for raw_clause in clauses_to_process:
                            clean_clause = raw_clause.strip()
                            if len(clean_clause) > 2:
                                clause_pcm = b""
                                for audio_chunk in voice.synthesize(clean_clause):
                                    clause_pcm += audio_chunk.audio_int16_bytes
                                
                                if clause_pcm:
                                    # Upscale to 44.1k and wrap inside an explicit WAV wrapper
                                    resampled_np = resample_22050_to_44100(clause_pcm)
                                    yield create_wav_chunk(resampled_np, sample_rate=44100)
                                await asyncio.sleep(0.001)

            if text_buffer.strip():
                clean_clause = text_buffer.strip()
                if len(clean_clause) > 2:
                    clause_pcm = b""
                    for audio_chunk in voice.synthesize(clean_clause):
                        clause_pcm += audio_chunk.audio_int16_bytes
                    if clause_pcm:
                        resampled_np = resample_22050_to_44100(clause_pcm)
                        yield create_wav_chunk(resampled_np, sample_rate=44100)
                    await asyncio.sleep(0.001)

        except Exception as e:
            print(f"\n[System Core Pipeline Failure]: {e}")
            yield b""

    # Swapped media_type to audio/wav so the phone knows exactly what container format is landing
    return StreamingResponse(audio_stream_generator(), media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)