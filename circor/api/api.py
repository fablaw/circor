from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from circor.interface.main import wav_to_2D
import librosa
from circor.interface.main import preprocess

app = FastAPI()
#app.state.model = print(wav_to_2D(compressed_data=True))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/predict')
def predict(patient_id: int = 2530) -> dict:


    X_pred = select_patients(patient_id)

    model = app.state.model
    X_processed = preprocess(X_pred)
    y_pred = model.predict(X_processed)

    if round(y_pred,2) >= 0.45:
        return f"There is a high probability that murmurs exist" #Heart conditions in general?
    return f"There is a high probability that murmurs do not exist"



    # yet to to code, remaining question : are we going with a recording or an image directly?
 #   return {'Should i go see the cardiologist ?': 'yes'}

@app.get("/")
def index():
    return {'is it working ?': 'Yes'}
