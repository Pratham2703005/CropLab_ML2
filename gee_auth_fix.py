"""
Alternative Google Earth Engine authentication with time drift handling
"""

import ee
import json
import os
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def initialize_earth_engine_with_time_handling():
    """
    Initialize Google Earth Engine with handling for time synchronization issues
    """
    service_account_path = 'earth-engine-service-account.json'
    
    try:
        # Check if already initialized
        try:
            ee.Number(1).getInfo()
            logger.info("✅ Google Earth Engine already initialized")
            return True
        except:
            pass

        if not os.path.exists(service_account_path):
            logger.error(f"❌ Service account file not found: {service_account_path}")
            return False

        with open(service_account_path, 'r') as f:
            service_account = json.load(f)

        logger.info(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")

        # Try the standard method first
        try:
            credentials = ee.ServiceAccountCredentials(
                email=service_account['client_email'],
                key_file=service_account_path
            )
            ee.Initialize(credentials)
            
            # Test the initialization
            test_result = ee.Number(1).getInfo()
            if test_result == 1:
                logger.info("✅ Google Earth Engine initialized successfully")
                return True
        except Exception as e:
            if "Invalid JWT Signature" in str(e):
                logger.warning("⚠️ JWT signature error detected - likely due to clock sync issues")
                logger.info("💡 Please synchronize your system clock and try again")
                logger.info("💡 Windows: Right-click clock → Adjust date/time → Sync now")
                logger.info("💡 Or run as admin: w32tm /resync")
                return False
            else:
                logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def check_time_sync():
    """
    Check if system time is properly synchronized
    """
    local_time = datetime.now()
    utc_time = datetime.utcnow()
    time_diff = abs((local_time - utc_time).total_seconds())
    
    logger.info(f"Local time: {local_time}")
    logger.info(f"UTC time: {utc_time}")
    logger.info(f"Time difference: {time_diff} seconds")
    
    if time_diff > 300:  # 5 minutes
        logger.warning(f"⚠️ Large time difference detected: {time_diff} seconds")
        logger.warning("This will cause JWT signature issues with Google Earth Engine")
        return False
    else:
        logger.info("✅ System time appears to be synchronized")
        return True