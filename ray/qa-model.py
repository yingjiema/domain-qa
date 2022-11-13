from starlette.requests import Request
import os

import ray
from ray import serve

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

from pydantic import BaseModel

# class Query(BaseModel):
#     text: str
#     index: str = 'bioasq'
#     embedding_model: str = "dmis-lab/biobert-base-cased-v1.2"
#     reader_model: str = "deepset/minilm-uncased-squad2"



@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2.0, "num_gpus": 1})
class QuestionAnswerer:
    def __init__(self):
        # Load model
        #self.document_store = FAISSDocumentStore.load(index_path="haystack_test_faiss", config_path="haystack_test_faiss_config")
        if os.path.exists('domain-qa-document-store-short.db'):
            os.remove('domain-qa-document-store-short.db')

        #corpus = ["Neutrophils are a type of white blood cell that protects us from bacteria.",
        #         "Lymphocytes are a type of white blood cell that protects us against viruses.",
        #         "Eosinophils are a type of white blood cell that fights parasites.",
        #         "T2DM, also known as type 2 diabetes mellitus, is a common condition" ]
        #self.document_store = FAISSDocumentStore(
        #    sql_url='sqlite:///domain-qa-document-store-short.db', 
        #    faiss_index_factory_str="Flat", 
        #    similarity="cosine")
        #self.document_store.write_documents([{"content": t} for t in corpus])
        self.document_store = ElasticsearchDocumentStore(
            host='es01',
            port='9200',
            username='',
            password='',
            index='bioasq',
            similarity="cosine",
            embedding_dim=768
        )
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
            model_format="sentence_transformers",
            max_seq_len=200,
            progress_bar=True
        )
        #self.document_store.update_embeddings(self.retriever)
        self.reader = FARMReader(
            model_name_or_path="deepset/minilm-uncased-squad2", 
            use_gpu=True)
        self.pipeline = ExtractiveQAPipeline(self.reader, self.retriever)

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    def predict(self, query):
        prediction = self.pipeline.run(
            query=query, 
            params={"Retriever": {"top_k": 20}, "Reader": {"top_k": 5}}
        )

        return prediction

    async def __call__(self, http_request: Request) -> dict:
        question: str = await http_request.json()
        pred = self.predict(question)

        return [x.to_dict() for x in pred["answers"]]


question_answerer = QuestionAnswerer.bind()
