import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 读取数据
data = pd.read_excel('附件数据及提交结果表格/附件表/附件1-商家历史出货量表.xlsx')

# 数据预处理
# 假设商家、仓库、商品等分类信息已编码为数值
# 将日期转换为日期时间对象
data['date'] = pd.to_datetime(data['date'])

# 将数据按商家、仓库、商品分组，以准备进行预测
grouped = data.groupby(['seller_no', 'warehouse_no', 'product_no'])

# 存储预测结果
result_table_1 = pd.DataFrame(columns=['seller_no', 'warehouse_no', 'product_no', 'date', 'predicted_demand'])

# 循环处理每个组
for group, group_data in grouped:
	seller, warehouse, product = group

	# 特征选择：这里可以根据实际情况选择需要的特征
	features = group_data[['date', 'qty']].copy()

	# 将日期设置为索引
	features.set_index('date', inplace=True)

	# 数据标准化
	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(features)

	# 划分训练集和测试集
	train_size = int(len(scaled_features) * 0.8)
	train_data, test_data = scaled_features[:train_size], scaled_features[train_size:]


	# 创建时间窗口数据
	def create_sequences(data, seq_length):
		sequences = []
		for i in range(len(data) - seq_length):
			seq = data[i:i + seq_length]
			sequences.append(seq)
		return np.array(sequences)


	seq_length = 10  # 可以根据数据和问题调整
	train_sequences = create_sequences(train_data, seq_length)
	test_sequences = create_sequences(test_data, seq_length)

	# 分割特征和标签
	X_train, y_train = train_sequences[:, :-1], train_sequences[:, -1]
	X_test, y_test = test_sequences[:, :-1], test_sequences[:, -1]

	# 构建LSTM模型
	model = Sequential()
	model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

	# 训练模型
	model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 使用模型进行预测
