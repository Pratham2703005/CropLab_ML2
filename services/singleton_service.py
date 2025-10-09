"""
Singleton service for loading and managing ML model and Google Earth Engine
"""
import tensorflow as tf
import joblib
import logging
import os
import threading
from typing import Optional
import merged_processor

logger = logging.getLogger(__name__)


class SingletonService:
    """Singleton class to manage ML model and GEE initialization"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._model = None
        self._scaler = None
        self._gee_initialized = False
        self._model_error = None
        self._scaler_error = None
        self._gee_error = None
        
        # Initialize everything
        self._load_model()
        self._load_scaler()
        self._initialize_gee()
        
        self._initialized = True
        logger.info("✅ SingletonService initialized successfully")
    
    def _load_model(self):
        """Load the ML model once"""
        try:
            if not os.path.exists("model.h5"):
                raise FileNotFoundError("model.h5 not found")
                
            # Try loading with different compatibility options
            try:
                self._model = tf.keras.models.load_model("model.h5", compile=False)
            except Exception as e1:
                logger.warning(f"First model load attempt failed: {e1}")
                # Try with custom objects if needed
                self._model = tf.keras.models.load_model(
                    "model.h5", 
                    compile=False, 
                    custom_objects=None
                )
            
            logger.info("✅ ML Model loaded successfully")
            
        except Exception as e:
            self._model_error = str(e)
            logger.error(f"❌ Error loading model: {e}")
            self._model = None
    
    def _load_scaler(self):
        """Load the scaler once"""
        try:
            if not os.path.exists("scaler.save"):
                raise FileNotFoundError("scaler.save not found")
                
            self._scaler = joblib.load("scaler.save")
            logger.info("✅ Scaler loaded successfully")
            
        except Exception as e:
            self._scaler_error = str(e)
            logger.error(f"❌ Error loading scaler: {e}")
            self._scaler = None
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine once"""
        try:
            success = merged_processor.initialize_earth_engine()
            if success:
                self._gee_initialized = True
                logger.info("✅ Google Earth Engine initialized successfully")
            else:
                self._gee_error = "GEE initialization returned False"
                logger.error("❌ Google Earth Engine initialization failed")
                
        except Exception as e:
            self._gee_error = str(e)
            logger.error(f"❌ Error initializing GEE: {e}")
            self._gee_initialized = False
    
    # Public methods to access loaded resources
    @property
    def model(self) -> Optional[tf.keras.Model]:
        """Get the loaded ML model"""
        return self._model
    
    @property
    def scaler(self):
        """Get the loaded scaler"""
        return self._scaler
    
    @property
    def is_gee_initialized(self) -> bool:
        """Check if GEE is initialized"""
        return self._gee_initialized
    
    @property
    def model_error(self) -> Optional[str]:
        """Get model loading error if any"""
        return self._model_error
    
    @property
    def scaler_error(self) -> Optional[str]:
        """Get scaler loading error if any"""
        return self._scaler_error
    
    @property
    def gee_error(self) -> Optional[str]:
        """Get GEE initialization error if any"""
        return self._gee_error
    
    def is_ready(self) -> bool:
        """Check if all services are ready"""
        return (
            self._model is not None and 
            self._scaler is not None and 
            self._gee_initialized
        )
    
    def get_status(self) -> dict:
        """Get detailed status of all services"""
        return {
            "model_loaded": self._model is not None,
            "scaler_loaded": self._scaler is not None,
            "gee_initialized": self._gee_initialized,
            "all_ready": self.is_ready(),
            "errors": {
                "model_error": self._model_error,
                "scaler_error": self._scaler_error,
                "gee_error": self._gee_error
            }
        }
    
    def force_reinitialize(self):
        """Force reinitialize all services (for debugging)"""
        logger.info("🔄 Force reinitializing all services...")
        self._model = None
        self._scaler = None
        self._gee_initialized = False
        self._model_error = None
        self._scaler_error = None
        self._gee_error = None
        
        self._load_model()
        self._load_scaler()
        self._initialize_gee()
        
        logger.info("🔄 Force reinitialization complete")


# Global instance getter
def get_singleton_service() -> SingletonService:
    """Get the singleton service instance"""
    return SingletonService()