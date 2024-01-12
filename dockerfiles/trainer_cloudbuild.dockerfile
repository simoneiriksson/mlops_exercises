# Base image
FROM python:3.10-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc
RUN apt install -y git 
RUN apt clean 
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /mlops_exercises
COPY mlops_exercises/ mlops_exercises/
#COPY data/ data/
#COPY reports/ reports/
#COPY models/ models/

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt  --no-cache-dir

RUN mkdir reports
RUN mkdir models
RUN mkdir reports/figures

#COPY .git .git
#RUN git init
#COPY .dvc .dvc
#COPY data.dvc data.dvc
#RUN dvc pull


####
##COPY .dvc .dvc
#COPY .dvc/config .dvc/config
#COPY data.dvc data.dvc
##COPY *.dvc ./dvc-folder
##RUN dvc init --no-scm
#RUN dvc config core.no_scm true
#RUN dvc pull --verbose

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc
RUN dvc config core.no_scm true
RUN dvc pull


ENTRYPOINT ["python", "-u", "mlops_exercises/train_model.py", "train"]
