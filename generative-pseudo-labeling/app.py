from fastapi import FastAPI, File, UploadFile

# from datasets import load_dataset
from generative_pseudo_label_generator import GenerativePseudoLabelGenerator
# from sentence_transformers import SentenceTransformer, util

def file_to_list(contents):
    contents = contents.decode('utf-8').split('\n')
    contents = [ c for c in contents if c != '' ]
    return contents


app = FastAPI()

# @app.post('/retriever-train', tags=['Generative Pseudo Labeling'])
# async def main(corpus, retriever_dir='retriever'):
#     gpl = GenerativePseudoLabelGenerator()
#     gpl.add_documents(corpus)
#     gpl.create_embedding_retriever()
#     gpl.train()
#     gpl.save(retriever_dir)

@app.post('/train-retriever', tags=['Generative Pseudo Labelling'])
async def train_retriever(file: UploadFile = File(...)):
    contents = file['text']

    sql_url = 'sqlite:///' + file['name'] + '_document_store.db'
    gpl = GenerativePseudoLabelGenerator(sql_url=sql_url)
    gpl.add_documents(contents)
    return {'messages': contents }

@app.get('/', tags=['Health Check'])
async def root():
    return {'message': 'Generative Pseudo Labeler service is up!'}
