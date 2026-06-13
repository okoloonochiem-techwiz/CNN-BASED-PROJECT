from flask import Flask, Response, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import threading
import time
from datetime import datetime
import BlynkLib
import random
import os

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\user\OneDrive\Desktop\potatoe prediction\potatoes_con.keras"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
BLYNK_AUTH = "f3z6V1VnA5ImX64N8PRxmjgaMh7Q5u5R"
CAPTURE_INTERVAL = 5  # seconds
PORT = 5000
CAMERA_INDEX = 0
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ---------------------------------------

# ---- Sensor thresholds ----
DROUGHT_TEMP_THRESHOLD = 35.0
PEST_HUM_THRESHOLD = 70.0
DRY_SOIL_THRESHOLD = 30.0
# ----------------------------

# Initialize
app = Flask(__name__)
model = load_model(MODEL_PATH, compile=False)
camera = cv2.VideoCapture(CAMERA_INDEX)
blynk = BlynkLib.Blynk(BLYNK_AUTH)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- Simulated Sensors ----
class FakeSensors:
    @property
    def temperature(self):
        return round(random.uniform(25.0, 45.0), 1)
    @property
    def humidity(self):
        return round(random.uniform(30.0, 90.0), 1)
    @property
    def moisture(self):
        return round(random.uniform(10.0, 100.0), 1)
sensors = FakeSensors()
# ----------------------------

# ---- Helper: prepare frame for CNN ----
def prepare_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img = np.expand_dims(img / 255.0, axis=0)
    return img

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
# ---------------------------------------

# ---- Flask video stream ----
def generate_frames():
    last_prediction_time = 0
    latest_label = "No prediction yet"
    latest_conf = 0.0

    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        # Every 5 seconds, run prediction
        if current_time - last_prediction_time >= CAPTURE_INTERVAL:
            img = prepare_frame(frame)
            preds = model.predict(img)
            label = CLASS_NAMES[np.argmax(preds)]
            conf = np.max(preds) * 100
            last_prediction_time = current_time
            latest_label, latest_conf = label, conf

            # --- Get sensor data ---
            temp = sensors.temperature
            hum = sensors.humidity
            moisture = sensors.moisture

            # --- Determine status ---
            status = "✅ Normal"
            if temp > DROUGHT_TEMP_THRESHOLD:
                status = "⚠️ Drought Risk!"
            if hum > PEST_HUM_THRESHOLD:
                status = "⚠️ Pest Risk!"
            if moisture < DRY_SOIL_THRESHOLD:
                status = "⚠️ Low Soil Moisture!"

            # Log result
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {label} ({conf:.1f}%) | Temp:{temp}°C | Hum:{hum}% | Moisture:{moisture}% | {status}")

            # Send to Blynk
            blynk.virtual_write(9, label)
            blynk.virtual_write(8, f"{conf:.1f}%")
            blynk.virtual_write(6, temp)
            blynk.virtual_write(2, hum)
            blynk.virtual_write(5, moisture)
            blynk.virtual_write(4, status)

        # Overlay prediction text on live stream
        cv2.putText(frame, f"{latest_label} ({latest_conf:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---- Flask routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No image selected for uploading'

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = prepare_image(filepath)
    preds = model.predict(img)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100
    result_text = f"Prediction: {pred_class} ({confidence:.2f}% confidence)"
    return render_template('index.html', prediction=result_text, img_path=filepath)
# ---------------------------------------

def run_flask():
    print(f"🌿 Flask stream running on: http://127.0.0.1:{PORT}/video_feed")
    app.run(host='0.0.0.0', port=PORT, threaded=True)

if __name__ == "__main__":
    t1 = threading.Thread(target=run_flask)
    t1.start()

    # Keep Blynk running in main thread
    while True:
        blynk.run()
        time.sleep(0.1)

camera.release()
