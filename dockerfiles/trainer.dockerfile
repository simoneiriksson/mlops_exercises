# Base image
FROM --platform=linux python:3.10-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc
#RUN apt install -y git 
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /mlops_exercises
COPY mlops_exercises/ mlops_exercises/
#COPY data/ data/
COPY reports/ reports/
COPY models/ models/

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt  --no-cache-dir

COPY .git .git
COPY .dvc .dvc
COPY data.dvc data.dvc
RUN dvc pull

ENTRYPOINT ["python", "-u", "mlops_exercises/train_model.py", "train"]
