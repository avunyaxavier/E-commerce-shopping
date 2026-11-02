
import joblib
import sklearn
import numpy as np

print("Scikit-learn:", sklearn.__version__)
print("NumPy:", np.__version__)

try:
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully!")
    print(model)

except Exception as e:
    print("Model loading error:")
    print(repr(e))
