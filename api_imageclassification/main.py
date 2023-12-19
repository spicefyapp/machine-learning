import json
from flask import Flask, request, jsonify
import io
from PIL import Image
from keras.models import load_model
from spicedetect import descriptions, predict_label
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((384, 384), Image.NEAREST)
    
    pred_label, pred_accuracy = predict_label(img)
    description = descriptions.get(pred_label, "Deskripsi tidak tersedia untuk jenis rempah ini.")
    
    return jsonify({
        "label": pred_label,
        "accuracy": f"{pred_accuracy:.2f}%",
        "description": description
    })

if __name__ == "__main__":
    app.run(debug=True)