from fastapi import FastAPI
from prediction import ImmothepPrediction
import json
import numpy as np

predictor = ImmothepPrediction()

app = FastAPI()

@app.get("/")
def welcome_message():
    return {"welcome_message": "World"}

@app.get("/api/estimate")
def estimate(metre_carre: float, nb_pieces: int, terrain: float, code_postal: int):
    model_appart = predictor.trainLinearLeRetourAPIAppart(code_postal)
    model_maison = predictor.trainLinearLeRetourAPIMaison(code_postal)

    estimation_appart = predictor.predictionLinearAPI(nb_pieces, metre_carre, terrain, model_appart)
    estimation_appart = np.around(estimation_appart.item(), decimals=2)
    estimation_appart = str(estimation_appart)

    estimation_maison = predictor.predictionLinearAPI(nb_pieces, metre_carre, terrain, model_maison)
    estimation_maison = np.around(estimation_maison.item(), decimals=2)
    estimation_maison = str(estimation_maison)
    return {"estimate_appartment": estimation_appart + " €", "estimate_house": estimation_maison + " €"}
