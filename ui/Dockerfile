FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
COPY ./pages ./pages
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
