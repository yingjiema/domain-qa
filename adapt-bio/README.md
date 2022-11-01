## Initialization
- **Elasticsearch Host**: `$ export ELASTICSEARCH_HOST={public ip address of ES host}`
- **MLflow UI**: `mlflow ui --port 8004 --host 0.0.0.0`
- **FastAPI**: `uvicorn api:app --host 0.0.0.0 --port 8000`