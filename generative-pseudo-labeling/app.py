from fastapi import FastAPI, File, UploadFile
import boto3
import json

# from datasets import load_dataset
from generative_pseudo_label_generator import GenerativePseudoLabelGenerator
# from sentence_transformers import SentenceTransformer, util

def file_to_list(contents):
    contents = contents.decode('utf-8').split('\n')
    contents = [ c for c in contents if c != '' ]
    return contents

BUCKET_NAME = 'jason-jones-learned-his-lesson'
s3 = boto3.client('s3')
app = FastAPI()

# @app.post('/retriever-train', tags=['Generative Pseudo Labeling'])
# async def main(corpus, retriever_dir='retriever'):
#     gpl = GenerativePseudoLabelGenerator()
#     gpl.add_documents(corpus)
#     gpl.create_embedding_retriever()
#     gpl.train()
#     gpl.save(retriever_dir)

@app.post('/train-retriever', tags=['Generative Pseudo Labelling'])
async def train_retriever(file_name_dict: dict):

    file_name = file_name_dict['file_name']
    proj_name = file_name_dict['proj_name']
    key = f'projects/{proj_name}/{file_name}'

    # sql_url = file_name_dict['sql_url']
    
    # sql_url = 'postgresql://jason:jason@localhost:5433'
    gpl = GenerativePseudoLabelGenerator(index=proj_name)

    s3.download_file(BUCKET_NAME, Key=key,Filename=file_name)

    fh = open(file_name)
    if file_name.endswith('.json'):
        contents = json.load(fh)
    else:
        contents = f.readlines()
        contents = [ c for c in contents if c != '\n' ]
    
    fh.close()

    gpl.add_documents(contents)
    gpl.create_embedding_retriever()
    gpl.train()
    return {'messages': contents}

@app.post('/store-retriever', tags=['Move Trained Retriever to S3'])
async def store_retriever(file_name_dict: dict):

    file_name = file_name_dict['file_name']
    proj_name = file_name_dict['proj_name']
    full_file_name = f'projects/{proj_name}/{file_name}'

    with open(file_name, 'r+b') as f:
        response = s3.upload_fileobj(f, BUCKET_NAME, full_file_name)
    

    return {'messages': response}

@app.get('/', tags=['Health Check'])
async def root():
    return {'message': 'Generative Pseudo Labeler service is up!'}
