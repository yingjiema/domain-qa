from fastapi import FastAPI, File, UploadFile
from pipeline import QAPipeline
import numpy as np
import io

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='FastAPI')

@app.post("/question-answer/", tags=["Domain-specific QA"])
async def question(
    query: str = '', 
    index: str = '', 
    embedding_model: str = "dmis-lab/biobert-base-cased-v1.2", 
    reader_model: str = "deepset/minilm-uncased-squad2"):
    pipeline = QAPipeline()
    pipeline.create_pipeline(index, embedding_model, reader_model)
    return {"answers": pipeline.predict(query)}


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}