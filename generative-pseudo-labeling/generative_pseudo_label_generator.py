from haystack.nodes.retriever import EmbeddingRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.label_generator import PseudoLabelGenerator
#import boto3
#import pathlib
#import os

class GenerativePseudoLabelGenerator(object):
    def __init__(self, sql_url='sqlite:///domain-qa-document-store.db'):
        self.document_store = FAISSDocumentStore(sql_url=sql_url, faiss_index_factory_str="Flat", similarity="cosine")
        self.retriever = None
        
    def add_documents(self, corpus):
        self.document_store.write_documents([{"content": t} for t in corpus])

    def create_embedding_retriever(self, 
                            embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
                            model_format="sentence_transformers",
                            max_seq_len=200,
                            progress_bar=True):
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=embedding_model,
            model_format=model_format,
            max_seq_len=max_seq_len,
            progress_bar=progress_bar
        )
        self.document_store.update_embeddings(self.retriever)

    def train(self, max_questions_per_document=10):
        self.question_producer = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1",
                                                    max_length=64,
                                                    split_length=128,
                                                    batch_size=32,
                                                    num_queries_per_doc=3,
                                                    )
        self.pseudo_label_generator = PseudoLabelGenerator(question_producer=self.question_producer,
                                                           retriever=self.retriever,
                                                           max_questions_per_document=max_questions_per_document,
                                                           batch_size=32,
                                                           top_k=10
                                                           )

        self.output, self.pip_id = self.pseudo_label_generator.run(documents=self.document_store.get_all_documents())

        self.retriever.train(self.output['gpl_labels'])

    def save(self, path):
        self.retriever.save(path)

    # def to_s3(self, bucket_name):
    #     s3 = boto3.resource("s3")
    #     files = os.listdir(".")
    #     for file in files:
    #         file_name = os.path.join(pathlib.Path(__file__).parent.resolve(), file)




    
        
