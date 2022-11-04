## Initialization
- **Elasticsearch Host**: `$ export ELASTICSEARCH_HOST='{public ip address of ES host}'`
- **MLflow UI**: `mlflow ui --port 8004 --host 0.0.0.0`
- **FastAPI**: `uvicorn api:app --host 0.0.0.0 --port 8000`
- **Testing locally**: 
    - `docker build -t adapt:1.0 .`
    - `docker run --gpus all --env-file .env -p8000:8000 adapt:1.0`