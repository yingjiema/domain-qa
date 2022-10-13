from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import io

#We instantiate a deeplab model with the location of the pretrained models
#https://github.com/tensorflow/models/tree/master/research/deeplab
model_path = './frozen_inference_graph.pb'
# model = DeepLabModel(model_path)

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Serverless Lambda FastAPI')


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}