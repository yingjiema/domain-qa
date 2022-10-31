from fastapi import FastAPI
from ingestion import DocumentIngestion
from retriever_adaption import DomainAdaptionPipeline
import os 


#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='FastAPI')

@app.post("/ingest/", tags=["Write domain documents into Elasticsearch"])
async def ingestion(bucket: str = 'domain-qa-system', key: str = '', index: str = ''):
    try:
        host = os.environ.get('ELASTICSEARCH_HOST')
    except:
        print("ELASTICSEARCH_HOST host does not set as env parameter.")
    ingest = DocumentIngestion()
    ingest.load_docs_s3(bucket, key)
    ingested = ingest.write_docs(index, host=host)
    return {"doc_counts": ingested}

@app.post("/adapt/", tags=["Retriever adaption with ingested domain documents"])
async def adaption(index: str = ''):
    try:
        host = os.environ.get('ELASTICSEARCH_HOST')
    except:
        print("ELASTICSEARCH_HOST host does not set as env parameter.")
    adapt = DomainAdaptionPipeline()
    adapt.init_docstore_retriever(index, host=host)
    adapt.generate_labels()
    adapt.train()
    return


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}