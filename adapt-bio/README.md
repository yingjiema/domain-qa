## APIs for Domain Adaption with PseudoLabelGenerator
- **Elasticsearch Host**: If deployed Elasticsearch in a different host, run `$ export ELASTICSEARCH_HOST='{public ip address of ES host}'` to set the environment variable.
- **MLflow UI**: `mlflow ui --port 8004 --host 0.0.0.0`
- **FastAPI**: `uvicorn api:app --host 0.0.0.0 --port 8000`
- **Local test**: To test with docker locally and enable GPUs.
    - `docker build -t adapt:1.0 .`
    - `docker run --gpus all --env-file .env -p8000:8000 adapt:1.0`