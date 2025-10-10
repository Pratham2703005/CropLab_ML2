# utils/preprocess.py

import numpy as np

def preprocess_input(ndvi_data, sensor_data, scaler):
    """
    Preprocess NDVI and sensor data for model input.
    
    Args:
        ndvi_data: NDVI array, typically (1, H, W, 1)
        sensor_data: Sensor array, typically (1, H, W, C)
        scaler: Fitted scaler for sensor data
    
    Returns:
        Tuple of (preprocessed_ndvi, preprocessed_sensor)
    """
    try:
        # Normalize NDVI data
        ndvi_data = ndvi_data / 255.0

        # Handle sensor data scaling
        if sensor_data.ndim == 5:
            # 5D case: (N, T, H, W, C)
            N, T, H, W, C = sensor_data.shape
            reshaped = sensor_data.reshape(N * T * H * W, C)
            scaled = scaler.transform(reshaped)
            sensor_data = scaled.reshape(N, T, H, W, C)
        elif sensor_data.ndim == 4:
            # 4D case: (N, H, W, C) - our current case
            N, H, W, C = sensor_data.shape
            reshaped = sensor_data.reshape(N * H * W, C)
            scaled = scaler.transform(reshaped)
            sensor_data = scaled.reshape(N, H, W, C)
        else:
            raise ValueError(f"Unsupported sensor data shape: {sensor_data.shape}. Expected 4D or 5D array.")

        return ndvi_data, sensor_data
        
    except Exception as e:
        print(f"Error in preprocess_input: {e}")
        print(f"NDVI shape: {ndvi_data.shape if ndvi_data is not None else 'None'}")
        print(f"Sensor shape: {sensor_data.shape if sensor_data is not None else 'None'}")
        raise
