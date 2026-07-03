import os
import re
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
from piper.voice import PiperVoice

app = FastAPI(title="Museo della Terra - Unified Voice Core")

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

    try:
        print(f"\n[User Question]: {interaction.user_input}")
        
        response = ollama.chat(
            model='smollm2:360m',
            messages=[
                {'role': 'system', 'content': artifact["system_prompt"]},
                {'role': 'user', 'content': interaction.user_input}
            ]
        )
        full_text = response['message']['content'].strip()
        print(f"[Luther Response]: {full_text}")

        if not full_text:
            raise ValueError("Ollama returned an empty response.")

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        with open(temp_wav_path, "wb") as wav_file:
            voice.synthesize(full_text, wav_file)
            
        print(f"[System Core]: Audio rendering complete. Serving file.")

        return FileResponse(
            path=temp_wav_path, 
            media_type="audio/wav", 
            filename="response.wav"
        )

    except Exception as e:
        print(f"[System Core Failure]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)