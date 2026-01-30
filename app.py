from flask import Flask, render_template, request, send_file
import os
import numpy as np
import csv
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- APP CONFIG ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/currency_model.h5"
LOG_FILE = "prediction_log.csv"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

# ---------------- UTILITY FUNCTIONS ----------------
def detect_denomination(filename):
    if filename.startswith("100"):
        return "₹100"
    elif filename.startswith("200"):
        return "₹200"
    elif filename.startswith("500"):
        return "₹500"
    return "Unknown"

def generate_pdf(result, confidence, denomination, image_path):
    pdf_path = "static/report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Fake Currency Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Result: {result}")
    c.drawString(50, 740, f"Confidence: {confidence}%")
    c.drawString(50, 720, f"Denomination: {denomination}")

    c.drawImage(image_path, 50, 480, width=300, height=160)

    c.drawString(50, 440, "Developed by: Vedant Kumkar")
    c.save()

    return pdf_path

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = denomination = image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Image preprocessing
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            confidence = round(float(prediction) * 100, 2)

            if prediction > 0.5:
                result = "FAKE CURRENCY"
            else:
                result = "REAL CURRENCY"

            denomination = detect_denomination(file.filename)
            image_url = file_path

            # -------- CSV LOGGING (UNICODE SAFE) --------
            if not os.path.exists(LOG_FILE):
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "result",
                        "confidence",
                        "denomination",
                        "image_name"
                    ])

            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    result,
                    confidence,
                    denomination,
                    file.filename
                ])

            generate_pdf(result, confidence, denomination, file_path)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        denomination=denomination,
        image_url=image_url
    )

# ---------------- PDF DOWNLOAD ----------------
@app.route("/download")
def download():
    return send_file("static/report.pdf", as_attachment=True)

# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin")
def admin():
    total = fake = real = 0
    denom_count = {"₹100": 0, "₹200": 0, "₹500": 0}

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if "FAKE" in row["result"]:
                    fake += 1
                else:
                    real += 1

                if row["denomination"] in denom_count:
                    denom_count[row["denomination"]] += 1

    most_common = max(denom_count, key=denom_count.get) if total > 0 else "N/A"

    return render_template(
        "admin.html",
        total=total,
        fake=fake,
        real=real,
        most_common=most_common
    )

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
   app.run(host="0.0.0.0", port=10000)

