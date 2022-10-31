import mlflow

class DomainAdaptionPipeline(object):
    def __init__(self):
        self.document_store = None
        self.retriever = None

    def init_docstore_retriever(
        self, index, 
        host='localhost', 
        username='', 
        password='', 
        similarity='cosine', 
        embedding_dim=768,
        embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b", 
        model_format="sentence_transformers",
        max_seq_len=256,
        progress_bar=True,
        ):
        from haystack.document_stores import ElasticsearchDocumentStore
        from haystack.nodes.retriever import EmbeddingRetriever

        self.document_store = ElasticsearchDocumentStore(
            host=host,
            username=username,
            password=password,
            index=index,
            similarity=similarity,
            embedding_dim=embedding_dim
        )

        self.retriever = EmbeddingRetriever(
            document_store=self.document_store, 
            embedding_model=embedding_model, 
            model_format=model_format,
            max_seq_len=max_seq_len,
            progress_bar=progress_bar
        )
        self.document_store.update_embeddings(self.retriever)

    def generate_labels(
        self,
        model_name_or_path="doc2query/msmarco-t5-base-v1",
        max_length=64,
        split_length=128,
        batch_size=32,
        num_queries_per_doc=3,
        max_questions_per_document=10,
        top_k=10,
        progress_bar=False):
        from haystack.nodes.question_generator import QuestionGenerator
        from haystack.nodes.label_generator import PseudoLabelGenerator


        question_producer = QuestionGenerator(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            split_length=split_length,
            batch_size=batch_size,
            num_queries_per_doc=num_queries_per_doc)
        self.question_producer_params = question_producer.__dict__.get('_component_config', {}).get('params', {})
        # experiment_name = "domain-adaption-question-generator"  
        # mlflow.set_experiment(experiment_name)
        # with mlflow.start_run() as run:
        #     mlflow.log_params(self.question_producer_params)

        psg = PseudoLabelGenerator(
            question_producer=question_producer,
            retriever=self.retriever,
            max_questions_per_document=max_questions_per_document,
            batch_size=batch_size,
            top_k=top_k,
            progress_bar=progress_bar)
        self.psg_params = psg.__dict__.get('_component_config', {}).get('params', {})

        output, pipe_id = psg.run(
            documents=self.document_store.get_all_documents()) 
        self.gpl_labels = output['gpl_labels']

    def train(self, index):
        experiment_name = "domain-adaption"  
        # s3_bucket = "s3://domain-qa-system/mlruns" 
        # mlflow.create_experiment(experiment_name, s3_bucket)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            self.retriever.train(self.gpl_labels, n_epochs=2, batch_size=32)
            self.retriever.save(f'saved_models/{index}')
            try:
                params = {
                    'document_score': self.document_store.__dict__.get('_component_config', {}).get('params', {}),
                    'retriever': self.retriever.__dict__.get('_component_config', {}).get('params', {}),
                    'question_generator': self.question_producer_params,
                    'pseudo_label_generator': self.psg_params
                }
                mlflow.log_params(params)
            except:
                pass
            mlflow.log_artifacts('saved_models')
            self.document_store.update_embeddings(self.retriever)
