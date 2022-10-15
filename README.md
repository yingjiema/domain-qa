# Synthetic Data Generation Pipeline for Domain-Specific Question Answering System

## Problem
- **Domain-specific chatbot**: Includes domain-specific jargon (clinical or biomedical) that can complicate response generation for queries.
- **Bot attacks or outliers detection**: Identification and filtering of irrelevant questions that would significantly save resources for the business.

## Hypothesis
- **Business value**: Virtual assistants in the form of chatbots are an integral part of almost all service-oriented businesses today. 
- **Market size**: The global chatbot market size was estimated at $430.9M in 2020 and is expected to reach $525.7M by the end of 2021.

## Domain-Specific Data & Models
- Biomedical: BioASQ -> BioBERT (Pre-trained model)
  - [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- Clinical: CliCR -> ?

## Components & Actions
1. [DocumentStore](https://docs.haystack.deepset.ai/docs/document_store): ElasticSearch sever invoked by retriever
   - [Install](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) Elasticsearch and then [start](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html) an instance.
   - Pull Docker image and running it
   - Initialize the Haystack ElasticsearchDocumentStore() object that will connect to initialized ES instance.
   - Build an API service to index formatted documents with haystack
       - [Input Format](https://docs.haystack.deepset.ai/docs/document_store#input-format)
   - Test indexing API service with domain-specific data
       - Biomedical docs
       - Clinical docs

2. **[GenerativeQAPipeline](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#generativeqapipeline)**
    - **[Retriever](https://docs.haystack.deepset.ai/docs/retriever)**: Given query (domain-specific question), retrieve most relevant candidates Documents
        - Initialize a Retriever by passing a DocumentStore as its argument
    - **[AnswerGenerator](https://docs.haystack.deepset.ai/docs/answer_generator)**: Reads a set of candidate documents and generates an answer to a question, word by word
        - Initialize a locally hosted AnswerGenerator
    - Build an API service which implement GenerativeQAPipeline with haystack
    - Test answer generator API with domain-specific question
3. **[PseudoLabelGenerator](https://docs.haystack.deepset.ai/docs/pseudo_label_generator)** [Optional]**:** unsupervised domain adaptation method for **training dense retrievers** (not for reader)
    - Build an API service with PseudoLabelGeneraor to create training data
    - Test data generator API service with any domain-specific data
4. **[Annotation Tool](https://docs.haystack.deepset.ai/docs/annotation)** [Optional]: Human labeling to generate labels for question answering
5. **QA Frontend**: Chatbot?

## Services [TBD]

TBD: To Be Deployed

1. ElasticSearch
2. FastAPIs
    - [POST] Indexing API: Upload a formatted document.txt, this document will be indexed into ElasticSearch
    - [POST] Answer generator API: Ask a new question, return the generated answers
    - [POST] Data generator API (optional)
3. QA Web APP
4. Nvidia Triton: model serving (if needed)
5. Grafana: Monitoring