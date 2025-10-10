# utils/preprocess.py

import numpy as np

def preprocess_input(ndvi_data, sensor_data, scaler):
    """
    Preprocess NDVI and sensor data for model input.
    Simple and robust preprocessing with fallbacks.
    """
    try:
        # Normalize NDVI data (simple normalization)
        if ndvi_data is not None and ndvi_data.max() > 1.0:
            ndvi_data = ndvi_data / 255.0
        
        # Handle sensor data scaling with fallback
        if sensor_data is not None and scaler is not None:
            try:
                original_shape = sensor_data.shape
                
                # Flatten for scaling, keeping only the feature dimension
                if sensor_data.ndim >= 2:
                    # Reshape to (samples, features) for scaler
                    features = sensor_data.shape[-1]  # Last dimension is features
                    flat_data = sensor_data.reshape(-1, features)
                    
                    # Apply scaler
                    scaled_flat = scaler.transform(flat_data)
                    
                    # Reshape back to original shape
                    sensor_data = scaled_flat.reshape(original_shape)
                    
            except Exception as scale_error:
                print(f"Scaling failed, using raw sensor data: {scale_error}")
                # Fallback: use raw sensor data without scaling
                pass
        
        return ndvi_data, sensor_data
        
    except Exception as e:
        print(f"Preprocessing error, using raw data: {e}")
        # Ultimate fallback: return data as-is
        return ndvi_data, sensor_data
