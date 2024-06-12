FROM python:3.7.3-stretch

RUN mkdir -p /leaf-model-api

WORKDIR /leaf-model-api

RUN pip install --no-cache-dir -U pip

COPY docker-requirements.txt .

RUN python -m pip install -r docker-requirements.txt

COPY . .

CMD ["python", "main.py"]
