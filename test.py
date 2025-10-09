import ee
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_PATH = 'earth-engine-service-account.json'

if not os.path.exists(SERVICE_ACCOUNT_PATH):
    logger.error(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
    raise SystemExit(1)

with open(SERVICE_ACCOUNT_PATH, 'r') as f:
    info = json.load(f)

email = info.get('client_email')
logger.info(f"Using service account: {email}")

try:
    creds = ee.ServiceAccountCredentials(email=email, key_file=SERVICE_ACCOUNT_PATH)
    ee.Initialize(creds)
    res = ee.Number(1).getInfo()
    logger.info(f"Earth Engine test result: {res}")
    print('OK' if res == 1 else 'FAILED')
except Exception as e:
    logger.exception("GEE init failed")
    print("ERROR:", e)