from tensorflow.keras.models import load_model

# --- Path to your old model ---
old_model_path = "potatoes.h5"

# --- Path to save the new Keras format model ---
new_model_path = r"C:\Users\user\OneDrive\Desktop\potatoes_con.keras"

# --- Load the old model safely (disable compilation) ---
model = load_model(old_model_path, compile=False)

# --- Save in new .keras format ---
model.save(new_model_path)

print("✅ Conversion successful!")
print(f"New model saved at: {new_model_path}")
