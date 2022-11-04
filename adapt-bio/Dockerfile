FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

WORKDIR /home/user

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install software dependencies
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        cmake \
        curl \
        git \
        libpoppler-cpp-dev \
        libtesseract-dev \
        pkg-config \
        poppler-utils \
        python3-pip \
        python3.7 \
        python3.7-dev \
        python3.7-distutils \
        swig \
        tesseract-ocr \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install PDF converter
RUN curl -s https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz | tar -xvzf - -C /usr/local/bin --strip-components=2 xpdf-tools-linux-4.04/bin64/pdftotext

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7


# Install package
RUN pip install --upgrade pip
RUN pip install 'farm-haystack[all]'

# Install PyTorch for CUDA 11
RUN pip3 install --no-cache-dir torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app","--host", "0.0.0.0"]
# CMD ["mlflow", "ui","--host", "0.0.0.0", "--port", "8004"]