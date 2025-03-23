# Chạy lệnh python train_ocr.py --az "A_Z Handwritten Data.csv" --model handwriting.keras --plot "plot.png"
# Kiểm tra và sử dụng GPU nếu có
import tensorflow as tf

# Kiểm tra xem có GPU không
if tf.config.list_physical_devices('GPU'):
    print("[INFO] GPU is available")
    device_name = tf.test.gpu_device_name()
    print(f"[INFO] Using GPU: {device_name}")
else:
    print("[WARNING] GPU is not available, using CPU instead")
    
# Đặt backend của matplotlib để lưu hình ảnh trong chế độ nền
import matplotlib
matplotlib.use("Agg")

from models import ResNet 
from dataset import load_mnist_dataset 
from dataset import load_az_dataset  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from imutils import build_montages  
import matplotlib.pyplot as plt  
import numpy as np  
import argparse 
import cv2  

# Khởi tạo trình phân tích tham số dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True,
    help="Đường dẫn đến tập dữ liệu A-Z")
ap.add_argument("-m", "--model", type=str, required=True,
    help="Đường dẫn để lưu mô hình sau khi huấn luyện")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="Đường dẫn để lưu biểu đồ lịch sử huấn luyện")
args = vars(ap.parse_args())

# Định nghĩa số epoch, learning rate ban đầu và batch size
EPOCHS = 50 
INIT_LR = 1e-1  
BS = 128 

# Load tập dữ liệu A-Z và MNIST
print("[INFO] loading datasets...")
(azData, azLabels) = load_az_dataset(args["az"])  # Load dữ liệu chữ cái A-Z
(digitsData, digitsLabels) = load_mnist_dataset()  # Load dữ liệu số 0-9

# Để tránh trùng nhãn, cộng thêm 10 vào nhãn của dữ liệu A-Z
azLabels += 10

# Ghép dữ liệu A-Z và MNIST vào một tập dữ liệu chung
data = np.vstack([azData, digitsData])  # Ghép ảnh
labels = np.hstack([azLabels, digitsLabels])  # Ghép nhãn

# Chuyển kích thước ảnh từ 28x28 thành 32x32 để phù hợp với mô hình
# Lặp qua từng ảnh và thay đổi kích thước
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# Thêm chiều kênh vào mỗi ảnh và chuẩn hóa pixel về khoảng [0,1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# Chuyển đổi nhãn thành dạng one-hot
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Xử lý mất cân bằng dữ liệu
classTotals = labels.sum(axis=0)
classWeight = {}
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Chia tập dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.20, stratify=labels, random_state=42)

# Khởi tạo bộ tạo dữ liệu tăng cường
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

# Khởi tạo và biên dịch mô hình
print("[INFO] compiling model...")
lr_schedule = ExponentialDecay(initial_learning_rate=INIT_LR, decay_steps=1000, decay_rate=0.96)
opt = SGD(learning_rate=lr_schedule)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
    (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# Huấn luyện mô hình
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)

# Định nghĩa danh sách nhãn
labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Đánh giá mô hình
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

# Lưu mô hình đã huấn luyện
print("[INFO] serializing network...")
model.save(args["model"], save_format="keras")

# Vẽ và lưu biểu đồ quá trình huấn luyện
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Hiển thị kết quả nhận dạng ngẫu nhiên
images = []
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0) if prediction[0] == np.argmax(testY[i]) else (0, 0, 255)

    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        color, 2)
    images.append(image)

# Tạo ảnh ghép từ các kết quả dự đoán
montage = build_montages(images, (96, 96), (7, 7))[0]

# Hiển thị kết quả
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)