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
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        swig \
        tesseract-ocr \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8


# Install package
RUN pip install --upgrade pip
RUN pip install 'farm-haystack[all-gpu]'
RUN pip install ray[serve]


COPY . .
CMD ["serve", "run", "qa-model:question_answerer", "--host", "0.0.0.0"]
# CMD ["mlflow", "ui","--host", "0.0.0.0", "--port", "8004"]
