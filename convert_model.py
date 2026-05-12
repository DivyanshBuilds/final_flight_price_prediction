from tensorflow.keras.models import load_model

model = load_model("models/ann_model.h5", compile=False)
model.save("models/ann_model.keras")  # Use .keras extension
print("Model saved successfully!")