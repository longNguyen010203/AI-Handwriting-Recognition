# Import các thư viện cần thiết
from tensorflow.keras.datasets import mnist  # Thư viện chứa tập dữ liệu MNIST
import numpy as np  # Thư viện xử lý dữ liệu số học

def load_az_dataset(datasetPath):
    # Khởi tạo danh sách để lưu dữ liệu ảnh và nhãn tương ứng
    data = []
    labels = []

    # Duyệt qua từng dòng dữ liệu trong file (dữ liệu được lưu theo dạng CSV)
    for row in open(datasetPath):
        # Chia nhỏ dòng dữ liệu thành danh sách bằng dấu phẩy ","
        row = row.split(",")

        # Phần tử đầu tiên là nhãn của chữ cái (từ 0 đến 25, ứng với A-Z)
        label = int(row[0])

        # Các phần còn lại là các giá trị pixel của ảnh (28x28 = 784 giá trị)
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        # Chuyển dữ liệu từ dạng vector 1D (784 giá trị) thành ma trận 2D (28x28)
        image = image.reshape((28, 28))

        # Lưu ảnh và nhãn vào danh sách
        data.append(image)
        labels.append(label)

    # Chuyển danh sách thành mảng NumPy để dễ xử lý với TensorFlow/Keras
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    # Trả về bộ dữ liệu dưới dạng tuple (data, labels)
    return (data, labels)

def load_mnist_dataset():
    # Tải tập dữ liệu MNIST, gồm 60.000 ảnh train và 10.000 ảnh test
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # Gộp toàn bộ dữ liệu train và test vào một mảng chung
    data = np.vstack([trainData, testData])  # Nối mảng ảnh
    labels = np.hstack([trainLabels, testLabels])  # Nối mảng nhãn

    # Trả về bộ dữ liệu dưới dạng tuple (data, labels)
    return (data, labels)