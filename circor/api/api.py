from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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


app.state.model = load_model()

@app.post('/predict/')
async def predict(file:UploadFile):
    temp=f'./circor/app/processed_wav/{file.filename}'
    y_pred = pred(X_pred=temp, model=app.state.model)

    return {
            "Prediction": y_pred
            }

@app.get("/")
def index():
    return {'is it working ?': 'Yes'}
