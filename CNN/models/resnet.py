# Import các lớp cần thiết từ TensorFlow/Keras
from tensorflow.keras.layers import BatchNormalization  # Chuẩn hóa batch giúp ổn định huấn luyện
from tensorflow.keras.layers import Conv2D  # Lớp tích chập 2D
from tensorflow.keras.layers import AveragePooling2D  # Lớp pooling trung bình
from tensorflow.keras.layers import MaxPooling2D  # Lớp pooling tối đa (không được dùng ở đây)
from tensorflow.keras.layers import ZeroPadding2D  # Thêm padding vào ảnh đầu vào
from tensorflow.keras.layers import Activation  # Hàm kích hoạt (ReLU, Softmax, ...)
from tensorflow.keras.layers import Dense  # Lớp fully-connected (FC)
from tensorflow.keras.layers import Flatten  # Chuyển từ tensor thành vector 1D
from tensorflow.keras.layers import Input  # Lớp đầu vào của mô hình
from tensorflow.keras.models import Model  # API để tạo mô hình Keras
from tensorflow.keras.layers import add  # Cộng hai tensor (dùng cho shortcut connection)
from tensorflow.keras.regularizers import l2  # Regularization L2 để giảm overfitting
from tensorflow.keras import backend as K  # Truy cập backend của Keras (TF hoặc Theano)

# Xây dựng lớp ResNet
class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		
		# Định nghĩa shortcut (đường tắt) ban đầu là đầu vào
		shortcut = data

		# Khối 1: BatchNorm -> ReLU -> 1x1 Convolution (giảm số kênh)
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

		# Khối 2: BatchNorm -> ReLU -> 3x3 Convolution (tích chập chính)
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

		# Khối 3: BatchNorm -> ReLU -> 1x1 Convolution (tăng số kênh lên K)
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

		# Nếu giảm kích thước (`red=True`), áp dụng convolution 1x1 lên shortcut
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

		# Cộng shortcut với đầu ra của khối residual
		x = add([conv3, shortcut])

		# Trả về đầu ra của module residual
		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters,
		reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
		
		# Xác định input shape với định dạng "channels last"
		inputShape = (height, width, depth)
		chanDim = -1  # Trục của kênh màu (cuối cùng)

		# Nếu backend là "channels first", hoán đổi trục của input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1  # Trục kênh màu ở vị trí đầu tiên

		# Khởi tạo đầu vào mô hình
		inputs = Input(shape=inputShape)

		# Áp dụng BatchNorm và Convolution 3x3 đầu tiên
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)
		x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

		# Vòng lặp qua từng stage (tầng của ResNet)
		for i in range(0, len(stages)):
			# Nếu là tầng đầu tiên, stride=1; ngược lại, stride=2 để giảm kích thước
			stride = (1, 1) if i == 0 else (2, 2)

			# Thêm residual module đầu tiên của stage (giảm kích thước nếu cần)
			x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

			# Thêm các residual module tiếp theo trong stage
			for j in range(0, stages[i] - 1):
				x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

		# Áp dụng BatchNorm -> ReLU -> Pooling trung bình
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)  # Pooling 8x8 trước khi đưa vào FC

		# Flatten dữ liệu thành vector 1D để đưa vào lớp fully-connected
		x = Flatten()(x)

		# Lớp đầu ra với softmax cho phân loại
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)

		# Tạo mô hình Keras với đầu vào `inputs` và đầu ra `x`
		model = Model(inputs, x, name="resnet")

		# Trả về mô hình đã hoàn thành
		return model
