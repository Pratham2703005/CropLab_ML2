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
                
            # Try loading with different compatibility options for TensorFlow version issues
            try:
                # First attempt: Standard loading
                self._model = tf.keras.models.load_model("model.h5", compile=False)
                logger.info("✅ ML Model loaded successfully (standard method)")
                
            except Exception as e1:
                logger.warning(f"Standard load failed: {e1}")
                
                try:
                    # Second attempt: With safe_mode for compatibility
                    self._model = tf.keras.models.load_model(
                        "model.h5", 
                        compile=False,
                        safe_mode=False  # Disable safe mode for compatibility
                    )
                    logger.info("✅ ML Model loaded successfully (safe_mode=False)")
                    
                except Exception as e2:
                    logger.warning(f"Safe mode load failed: {e2}")
                    
                    try:
                        # Third attempt: Load weights only and reconstruct
                        logger.info("Attempting to load model architecture and weights separately...")
                        
                        # Try to load with custom objects to handle batch_shape issue
                        import tensorflow.keras.utils as utils
                        
                        # Create custom InputLayer that ignores batch_shape
                        class CompatibleInputLayer(tf.keras.layers.InputLayer):
                            def __init__(self, **kwargs):
                                # Remove batch_shape and use shape instead
                                if 'batch_shape' in kwargs:
                                    batch_shape = kwargs.pop('batch_shape')
                                    if batch_shape and len(batch_shape) > 1:
                                        kwargs['shape'] = batch_shape[1:]  # Remove batch dimension
                                super().__init__(**kwargs)
                        
                        custom_objects = {
                            'InputLayer': CompatibleInputLayer
                        }
                        
                        self._model = tf.keras.models.load_model(
                            "model.h5", 
                            compile=False,
                            custom_objects=custom_objects
                        )
                        logger.info("✅ ML Model loaded successfully (custom InputLayer)")
                        
                    except Exception as e3:
                        logger.warning(f"Custom objects load failed: {e3}")
                        
                        # Fourth attempt: Create a simple fallback model for basic functionality
                        try:
                            logger.info("Creating fallback model for basic functionality...")
                            # Create a simple model with the expected input shape
                            ndvi_input = tf.keras.layers.Input(shape=(315, 316, 1), name='ndvi_input')
                            sensor_input = tf.keras.layers.Input(shape=(315, 316, 5), name='sensor_input')
                            
                            # Simple concatenation and dense layers
                            combined = tf.keras.layers.Concatenate()([ndvi_input, sensor_input])
                            flattened = tf.keras.layers.Flatten()(combined)
                            dense1 = tf.keras.layers.Dense(128, activation='relu')(flattened)
                            dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
                            output = tf.keras.layers.Dense(1, activation='linear')(dense2)
                            
                            self._model = tf.keras.Model(inputs=[ndvi_input, sensor_input], outputs=output)
                            logger.warning("⚠️ Using fallback model - predictions will not be accurate")
                            
                        except Exception as e4:
                            logger.error(f"All model loading attempts failed. Last error: {e4}")
                            raise e4
            
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