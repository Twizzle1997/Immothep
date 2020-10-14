from fastapi import FastAPI
import requests

app = FastAPI()

metre_carre = 740
nb_pieces = 4
terrain = 1000
code_postal = 63000

estimate = 'la fonction qui va faire l\'estimation'
request={"metre_carre" : metre_carre, "nb_pieces" : nb_pieces, "terrain" : terrain, "code_postal" : code_postal}

@app.get("/")
def welcome_message():
    return {"welcome_message": "World"}

@app.get("api/estimate/{request}")
def estimate(metre_carre: int, nb_pieces: int, terrain: int, code_postal: int):
    return {"estimate": estimate + nb_pieces}

    # uvicorn main:app --host 0.0.0.0 --port 80

url = "http://localhost:80/api/estimate/"
response = requests.get(url, params = request)
print(response.json())