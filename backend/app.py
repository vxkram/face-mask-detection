import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "mask_detector.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
IMAGE_SIZE = (160, 160)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

_model = None
_class_names = None


class ModelNotFoundError(RuntimeError):
    pass


def get_model():
    global _model, _class_names
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise ModelNotFoundError(
                "Model not found. Run `python train.py` first — see the README."
            )
        import tensorflow as tf  # imported lazily so the app can start fast and report a clear error

        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
    return _model, _class_names


def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.content_type.startswith("image/"):
        return jsonify({"error": "Only image files are allowed."}), 400

    try:
        model, class_names = get_model()
    except ModelNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        batch = preprocess(file.read())
    except Exception:
        return jsonify({"error": "Could not read that image."}), 400

    probabilities = model.predict(batch, verbose=0)[0]
    ranked = sorted(zip(class_names, probabilities.tolist()), key=lambda kv: kv[1], reverse=True)

    return jsonify({
        "label": ranked[0][0],
        "confidence": ranked[0][1],
        "probabilities": {name: prob for name, prob in ranked},
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
