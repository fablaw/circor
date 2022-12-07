from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from circor.interface.main import pred
from circor.ml_logics.registry import load_model

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_cnn = load_model()


@app.post('/predict')
async def process_audio(audio_file) -> float:

    audio = AudioSegment.from_file(audio_file)
    y_pred = pred(X_pred= audio, model=model_cnn)

    return {'probability to have a murmur is': y_pred}

@app.get("/")
def index():
    return {'is it working ?': 'Yes'}
