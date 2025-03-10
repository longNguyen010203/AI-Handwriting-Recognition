import base64
import io
from flask_cors import CORS
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)

# Load mô hình
model = tf.keras.models.load_model("model/my_model.keras")


def process_base64_image(base64_str):
    """ Xử lý ảnh từ base64 để đưa vào mô hình """

    try:
        # Giải mã base64 thành ảnh
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Lưu ảnh gốc để kiểm tra
        image.save("received_image.png")

        # Chuyển ảnh sang numpy array
        image = np.array(image)

        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Lấy histogram để xác định màu nền
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        most_common_intensity = np.argmax(histogram)

        if most_common_intensity > 128:
            # Ảnh nền trắng, chữ đen (ảnh viết tay trên giấy)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            inverted = cv2.bitwise_not(binary)

            # Làm dày chữ bằng Morphological Dilation
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.dilate(inverted, kernel, iterations=2)
        else:
            # Ảnh nền đen, chữ trắng
            processed = gray

        # Resize về 28x28
        processed = cv2.resize(processed, (28, 28), interpolation=cv2.INTER_AREA)

        # Lưu ảnh đã xử lý để kiểm tra
        cv2.imwrite("processed_image.png", processed)

        # Chuyển đổi sang tensor để đưa vào mô hình
        processed = processed.astype(np.float32) / 255.0  # Chuẩn hóa về [0,1]
        processed = np.expand_dims(processed, axis=-1)  # Thêm channel (28,28) -> (28,28,1)
        processed = np.expand_dims(processed, axis=0)   # Thêm batch dimension (1,28,28,1)

        return tf.convert_to_tensor(processed)

    except Exception as e:
        print("❌ Lỗi khi xử lý ảnh:", str(e))
        return None


@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/recognize")
def recognize():
    return render_template("recognize.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    label_map = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',
                 9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',
                 17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
    
    try:
        # Lấy dữ liệu base64 từ request JSON
        data = request.get_json()
        base64_str = data.get("image", "")

        print("📸 Nhận ảnh base64 (50 ký tự đầu):", base64_str[:50])  # Debug input

        if not base64_str:
            return jsonify({"error": "No image provided"}), 400

        # Xử lý ảnh đầu vào
        processed_tensor = process_base64_image(base64_str)

        if processed_tensor is None:
            return jsonify({"error": "Error in image processing"}), 500

        print("✅ Hình dạng tensor đầu vào:", processed_tensor.shape)  # Debug shape tensor

        # Dự đoán với mô hình
        try:
            predictions = model.predict(processed_tensor)
            predicted_label = predictions.argmax(axis=1)[0]
            print("🔍 Dự đoán:", predicted_label)  # Debug kết quả

            return jsonify({"prediction": label_map[int(predicted_label)]})

        except Exception as e:
            print("❌ Lỗi khi dự đoán:", str(e))
            return jsonify({"error": "Error during prediction"}), 500

    except Exception as e:
        print("❌ Lỗi hệ thống:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
