from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from circor.interface.main import wav_to_2D


app = FastAPI()
#app.state.model = print(wav_to_2D(compressed_data=True))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get('/show')
def audio_to_spectro():
    spectrogram = 'anna'
    return { 'lulu' : spectrogram}


@app.get('/predict')
def predict():
    # yet to to code, remaining question : are we going with a recording or an image directly?
    return {'Should i go see the cardiologist ?': 'yes'}

@app.get("/")
def index():
    return {'is it working ?': 'Yes'}
