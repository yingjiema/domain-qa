from fastapi import FastAPI, File, UploadFile
from pipeline import QAPipeline
import numpy as np
import io
from pydantic import BaseModel

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='FastAPI')

# pydantic classes
class SubmittedText(BaseModel):
    text: str


@app.post("/question-answer", tags=["Biomedical QA"])
async def question(query: SubmittedText):
    print('query:')
    print(query.text)
    pipeline = QAPipeline()
    pipeline.create_pipeline()
    return {"answers": pipeline.predict(query.text)}


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}