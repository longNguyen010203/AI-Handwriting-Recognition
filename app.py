import base64
import io
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf



app = Flask(__name__)


model = tf.keras.models.load_model("your_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        # Nhận ảnh từ request
        data = request.json
        image_data = data["image"].split(",")[1]  # Loại bỏ header của base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Chuyển về ảnh grayscale

        # Chuyển ảnh thành numpy array
        image = np.array(image)

        # Tiền xử lý ảnh (resize về kích thước model cần, ví dụ 28x28)
        image = cv2.resize(image, (28, 28))
        image = image / 255.0  # Chuẩn hóa pixel về [0,1]
        image = image.reshape(1, 28, 28, 1)  # Định dạng đầu vào cho model CNN

        # Dự đoán bằng model AI
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        return jsonify({"text": str(predicted_class)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)