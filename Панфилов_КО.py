# Импорт библиотек
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# Импорт данных
data = pd.read_csv('data/data_stocks.csv')

# Сброс переменной date
data = data.drop(columns=['DATE'])

# Размерность датасета
n = data.shape[0]
p = data.shape[1]

# Формирование данных в numpy-массив
data = data.values

# Данные для тестирования и обучения
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Шкалирование данных
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Построение X и y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Параметры архитектуры модели
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Плейсхолдер
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

# Инициализаторы
sigma = 1
weight_initializer = tf.keras.initializers.VarianceScaling(scale=sigma, mode='fan_avg', distribution='uniform', seed=0)
bias_initializer = tf.keras.initializers.Zeros()

# Уровень 1: Переменные для скрытых весов и смещений
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Уровень 2: Переменные для скрытых весов и смещений
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Уровень 3: Переменные для скрытых весов и смещений
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Уровень 4: Переменные для скрытых весов и смещений
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Уровень выходных данных: Переменные для скрытых весов и смещений
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Скрытый уровень
hidden_1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.linalg.matmul(X, W_hidden_1), bias_hidden_1), alpha=0.01)
hidden_2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.linalg.matmul(hidden_1, W_hidden_2), bias_hidden_2), alpha=0.01)
hidden_3 = tf.nn.leaky_relu(tf.nn.bias_add(tf.linalg.matmul(hidden_2, W_hidden_3), bias_hidden_3), alpha=0.01)
hidden_4 = tf.nn.leaky_relu(tf.nn.bias_add(tf.linalg.matmul(hidden_3, W_hidden_4), bias_hidden_4), alpha=0.01)

# Выходной уровень (должен быть транспонирован)
out = tf.transpose(tf.nn.bias_add(tf.linalg.matmul(hidden_4, W_out), bias_out))

# Функция стоимости
mse = tf.math.reduce_mean(tf.math.squared_difference(out, Y))

# Оптимизатор
opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

# Инициализация графа
net = tf.compat.v1.Session()

# Запуск инициализатора
net.run(tf.compat.v1.global_variables_initializer())

# Настройка интерактивного графика
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Количество эпох и размер сэмпла данных
epochs = 10
batch_size = 256
mse_train = []
mse_test = []

for e in range(epochs):

    # Перемешивание данных для обучения
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Обучение минипартией
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        # Запустить оптимизатор пакетов
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Показать прогресс
        if np.mod(i, 5) == 0:
            # Среднеквадратичная ошибка обучающей и тестовой выборки
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Предсказания
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            #plt.savefig(file_name)
            plt.pause(0.01)

# Вывести финальную фукнцию MSE после обучения
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)