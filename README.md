# Synthetic Data Generation Pipeline for Domain-Specific Question Answering System

## Problem
- **Domain-specific chatbot**: Includes domain-specific jargon (clinical or biomedical) that can complicate response generation for queries.
- **Bot attacks or outliers detection**: Identification and filtering of irrelevant questions that would significantly save resources for the business.

## Hypothesis
- **Business value**: Virtual assistants in the form of chatbots are an integral part of almost all service-oriented businesses today. 
- **Market size**: The global chatbot market size was estimated at $430.9M in 2020 and is expected to reach $525.7M by the end of 2021.

## Domain-Specific Data & Models
- Biomedical: BioASQ
- Clinical: CliCR
- COVID-19: TREC-COVID
- Pre-trained retriever: [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2?text=Paris+is+the+%5BMASK%5D+of+France.)
- Pre-trained reader: [doc2query/msmarco-t5-base-v1](https://huggingface.co/doc2query/msmarco-t5-base-v1?text=Python+is+an+interpreted%2C+high-level+and+general-purpose+programming+language.+Python%27s+design+philosophy+emphasizes+code+readability+with+its+notable+use+of+significant+whitespace.+Its+language+constructs+and+object-oriented+approach+aim+to+help+programmers+write+clear%2C+logical+code+for+small+and+large-scale+projects.)

## **Deployment**
- Create EC2 instance
- Pick deep learning ami
- Pick instance type: At least p3.2xlarge and 128G memory
- Create `./adapt-bio/.env` with the following parameters:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_access_key
AWS_DEFAULT_REGION=your_region
ELASTICSEARCH_HOST=es01
MLFLOW_TRACKING_URI=https://dagshub.com/domainqa/domain-qa.mlflow
MLFLOW_TRACKING_USERNAME=change_to_your_dagshub_username
MLFLOW_TRACKING_PASSWORD=change_to_your_dagshub_password
```
- Create `./env` with the following parameters:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_access_key
AWS_DEFAULT_REGION=your_region
ELASTICSEARCH_HOST=es01
```
- `docker-compose up --build`

## **Containers**
1. **ElasticSearch (es01)**
    - Document store for
    - [Setup Elasticsearch Cluster on AWS EC2](https://rharshad.com/setup-elasticsearch-cluster-aws-ec2/)
    - [Install](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) Elasticsearch and then [start](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html) an instance. Pull Docker image and running it: [Start a single-node cluster with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#docker-cli-run-dev-mode)
    - Initialize the Haystack ElasticsearchDocumentStore() object that will connect to initialized ES instance.
    - Build document ingestion API service to index formatted documents with haystack
        - [Input Format](https://docs.haystack.deepset.ai/docs/document_store#input-format)
    - Test indexing API service with domain-specific data
        - Biomedical docs
        - Clinical docs
    - Kibana UI
    - [Optional] Snapshot and Restore ElasticSearch indeices and clusters into [S3 repository](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/repository-s3.html)
        - [Tutorial](https://www.youtube.com/watch?v=1ZENLfY6Kkk)
        - [Kibana UI with Docker](https://www.elastic.co/guide/en/kibana/7.9/docker.html#_remove_docker_containers)
2. **[PseudoLabelGenerator](https://docs.haystack.deepset.ai/docs/pseudo_label_generator) (gpl)**
    - Unsupervised domain adaptation method for **training dense retrievers**
    - Build an API service with PseudoLabelGeneraor to train retriever with ingested domain specific document.
    - Experiment tracking with MLflow
    - Test domain adaption API service with any domain-specific data
3. **ExtractiveQAPipeline (demo)**
    - Build question answering API service
    - Able to select different domain for answering the given question.
4. **Model serving (ray)**
5. **QA Frontend (ui)**
    - Streamlit UI
    - Upload domain-specific document file to S3
    - UI is integrated with FastAPI services.
    - Question answering interface
6. **[Annotation Tool](https://docs.haystack.deepset.ai/docs/annotation)** [Future work]: Human labeling to generate labels for question answering
