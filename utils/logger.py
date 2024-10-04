from dotenv import load_dotenv
from google.cloud import logging as cloud_logging

if not load_dotenv():
    load_dotenv("/etc/secrets/env/.env")

_client = cloud_logging.Client()
logger = _client.logger("my_rag")
