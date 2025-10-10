"""
Singleton service for loading and managing ML model and Google Earth Engine
"""
import tensorflow as tf
import joblib
import logging
import os
import threading
import gc
from typing import Optional
import merged_processor

# Configure TensorFlow for memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
tf.config.experimental.enable_memory_growth = True

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
        
        # Force garbage collection after initialization to free up memory
        gc.collect()
        
        self._initialized = True
        logger.info("✅ SingletonService initialized successfully")
    
    def _load_model(self):
        """Load the ML model once with memory optimization"""
        try:
            if not os.path.exists("model.h5"):
                raise FileNotFoundError("model.h5 not found")
                
            # Configure TensorFlow for lower memory usage
            tf.config.experimental.enable_memory_growth = True
            
            # Try loading with different compatibility options for TensorFlow version issues
            try:
                # First attempt: Standard loading with memory optimization
                self._model = tf.keras.models.load_model("model.h5", compile=False)
                logger.info("✅ ML Model loaded successfully (standard method)")
            except Exception as e1:
                if "batch_shape" in str(e1):
                    logger.warning(f"TensorFlow version compatibility issue with batch_shape: {e1}")
                    try:
                        import h5py
                        import json
                        import tempfile
                        import shutil
                        # Create a temporary copy of the model file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                            shutil.copy2("model.h5", tmp_file.name)
                            temp_model_path = tmp_file.name
                        try:
                            # Read and modify the model config to robustly fix all batch_shape keys
                            with h5py.File(temp_model_path, 'r+') as f:
                                if 'model_config' in f.attrs:
                                    config_str = f.attrs['model_config']
                                    if isinstance(config_str, bytes):
                                        config_str = config_str.decode('utf-8')
                                    config = json.loads(config_str)
                                    # Recursively remove/convert all batch_shape keys
                                    def fix_all_batch_shape(obj):
                                        if isinstance(obj, dict):
                                            if 'batch_shape' in obj:
                                                batch_shape = obj['batch_shape']
                                                if batch_shape and isinstance(batch_shape, list) and len(batch_shape) > 1:
                                                    obj['shape'] = batch_shape[1:]
                                                del obj['batch_shape']
                                            for k, v in obj.items():
                                                fix_all_batch_shape(v)
                                        elif isinstance(obj, list):
                                            for item in obj:
                                                fix_all_batch_shape(item)
                                    fix_all_batch_shape(config)
                                    f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                            # Now try to load the modified model
                            self._model = tf.keras.models.load_model(temp_model_path, compile=False)
                            logger.info("✅ ML Model loaded successfully (all batch_shape keys fixed)")
                        finally:
                            if os.path.exists(temp_model_path):
                                os.unlink(temp_model_path)
                    except Exception as e2:
                        logger.warning(f"batch_shape fix failed: {e2}")
                        raise Exception(f"Model loading failed due to TensorFlow compatibility: {e1}")
                else:
                    logger.warning(f"Standard load failed: {e1}")
                    raise e1
            
        except Exception as e:
            self._model_error = str(e)
            logger.error(f"❌ Error loading model: {e}")
            self._model = None
        finally:
            # Clean up memory after model loading attempt
            gc.collect()
    
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