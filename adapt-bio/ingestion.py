class DocumentIngestion(object):
    def __init__(self):
        self.documents = []
    
    def load_docs_s3(self, bucket, key):
        import boto3
        import json

        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        self.documents = json.load(obj.get()['Body']) 

    def write_docs(self, index, host='localhost', username='', password=''):
        import requests
        from haystack.document_stores import ElasticsearchDocumentStore

        doc_store = ElasticsearchDocumentStore(
            host=host,
            username=username,
            password=password,
            index=index
        )
        doc_store.write_documents(self.documents)
        return requests.get(f'http://{host}:9200/{index}/_count').json()


