import os
import re
import asyncio
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
                
                # Split at clauses (commas, semicolons, ends of sentences) to feed audio quicker
                if any(pause in text_token for pause in [",", ";", ".", "!", "?", "\n"]):
                    clauses = re.split(r'(?<=[,;.!?\n])\s+', text_buffer)
                    clauses_to_process = clauses[:-1]
                    text_buffer = clauses[-1]

                    for raw_sentence in clauses_to_process:
                        clean_sentence = raw_sentence.strip()
                        if len(clean_sentence) > 2:
                            # Cleaned: Removed length_scale to stop the crash
                            for audio_chunk in voice.synthesize(clean_sentence):
                                yield audio_chunk.audio_int16_bytes
                            await asyncio.sleep(0.001)

            if text_buffer.strip():
                clean_sentence = text_buffer.strip()
                if len(clean_sentence) > 2:
                    # Cleaned: Removed length_scale here too
                    for audio_chunk in voice.synthesize(clean_sentence):
                        yield audio_chunk.audio_int16_bytes
                    await asyncio.sleep(0.001)

        except Exception as e:
            print(f"\n[System Core Pipeline Failure]: {e}")
            yield b""

    return StreamingResponse(audio_stream_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)