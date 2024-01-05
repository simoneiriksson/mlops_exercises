# Base image
FROM --platform=linux/arm64/v8 python:3.10-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_exercises/ mlops_exercises/
COPY data/ data/
COPY reports/ reports/


WORKDIR /
RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_exercises/train_model.py"]
