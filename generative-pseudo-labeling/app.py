from fastapi import FastAPI, File, UploadFile
import boto3

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

    sql_url = 'sqlite:///temp_document_store.db'
    # sql_url = 'postgresql://jason:jason@localhost:5433'
    gpl = GenerativePseudoLabelGenerator(sql_url=sql_url)

    s3.download_file(BUCKET_NAME, Key='projects/testing123/' + file_name,Filename=file_name)
    with open(file_name) as f:
        contents = f.readlines()

    contents = [ c for c in contents if c != '\n' ]
    gpl.add_documents(contents)
    gpl.create_embedding_retriever()
    gpl.train()
    return {'messages': contents}

@app.get('/', tags=['Health Check'])
async def root():
    return {'message': 'Generative Pseudo Labeler service is up!'}
