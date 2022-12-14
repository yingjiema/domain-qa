from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

class QAPipeline(object):

    def create_pipeline(self, index, embedding_model, reader_model):
        document_store = ElasticsearchDocumentStore(
            host='es01',
            port='9200',
            username='',
            password='',
            index=index,
            similarity="cosine",
            embedding_dim=768
        )
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=embedding_model,
        )
        reader = FARMReader(model_name_or_path=reader_model, use_gpu=True)
        pipe = ExtractiveQAPipeline(reader, retriever)
        self.pipeline = pipe

    def predict(self, query):
        prediction = self.pipeline.run(
            query=query, 
            params={"Retriever": {"top_k": 20}, "Reader": {"top_k": 5}}
        )
        return prediction

