FROM python:3.12.6-slim-bookworm

RUN apt-get -y update

WORKDIR /code

COPY ./requirements.txt /code/

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

EXPOSE 8080

ENV PYTHONUNBUFFERED True

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

