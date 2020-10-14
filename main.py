from typing import Optional
from fastapi import FastAPI, Form

app = FastAPI()
estimate = 'coucou'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/estimate/")
async def estimate(metre_carre: int = Form(...), nb_pieces: int = Form(...), terrain: int = Form(...), code_postal: int = Form(...)):
    return {"estimate": estimate}