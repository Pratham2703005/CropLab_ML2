# Compatibility Test Script for Deployment
# This script checks if all components work together properly

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} imported successfully")
except Exception as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully")
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
except Exception as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__} imported successfully")
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn
    print(f"✅ Uvicorn imported successfully")
except Exception as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    import gunicorn
    print(f"✅ Gunicorn imported successfully")
except Exception as e:
    print(f"❌ Gunicorn import failed: {e}")

try:
    import ee
    print(f"✅ Google Earth Engine imported successfully")
except Exception as e:
    print(f"❌ Google Earth Engine import failed: {e}")

# Test dependency compatibility
print("\n=== Testing Version Compatibility ===")

# Check numpy version compatibility with TensorFlow 2.13.0
numpy_version = np.__version__
tf_version = tf.__version__
sklearn_version = sklearn.__version__

print(f"NumPy: {numpy_version} (Required: >=1.22,<=1.24.3 for TF 2.13.0)")
print(f"TensorFlow: {tf_version}")
print(f"Scikit-learn: {sklearn_version} (Required: 1.2.2 for scaler compatibility)")

# Check if versions are compatible
numpy_major, numpy_minor, numpy_patch = map(int, numpy_version.split('.'))
tf_major, tf_minor, tf_patch = map(int, tf_version.split('.'))
sklearn_major, sklearn_minor, sklearn_patch = map(int, sklearn_version.split('.'))

if numpy_major == 1 and numpy_minor == 24 and numpy_patch <= 3:
    print("✅ NumPy version is compatible with TensorFlow 2.13.0")
else:
    print("❌ NumPy version may not be compatible with TensorFlow 2.13.0")

if tf_major == 2 and tf_minor == 13:
    print("✅ TensorFlow version is correct")
else:
    print("❌ TensorFlow version mismatch")

if sklearn_major == 1 and sklearn_minor == 2 and sklearn_patch == 2:
    print("✅ Scikit-learn version matches scaler (1.2.2)")
else:
    print("❌ Scikit-learn version mismatch - may cause scaler loading issues")

print("\n=== All checks completed ===")