# Base image
FROM --platform=linux python:3.10-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_exercises/ mlops_exercises/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_exercises/train_model.py", "train"]
