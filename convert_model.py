from tensorflow.keras.models import load_model

model = load_model("models/ann_model.h5", compile=False)
model.export("models/ann_model")  # Use export() instead of save()
print("Model exported successfully!")