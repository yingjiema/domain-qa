from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from pipeline import QAPipeline
import numpy as np
import io

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='FastAPI')

@app.post("/question-answer/{query}", tags=["Biomedical QA"])
async def question(query: str = ''):
    pipeline = QAPipeline()
    pipeline.create_pipeline()
    return {"answers": pipeline.predict(query)}


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}