import numpy as np
import pickle
import sys
import os
import tarfile
from six.moves.urllib.request import urlretrieve
import time

np.random.seed(0)

# Глобальные параметры
cifar_file_name = "cifar-10-python.tar.gz"
url = 'https://www.cs.toronto.edu/~kriz/'
data_root = 'D://ГУАП/Магистратура/2-ой семестр/Методы машинного обучения (Веселов)/Lab_1_Remastered/Dataset'
cifar_dir_name = "D://ГУАП/Магистратура/2-ой семестр/Методы машинного обучения (Веселов)/Lab_1_Remastered/Dataset"
# Обучение
alpha = 0.2  # Рейтинг обучения
num_epochs = 500  # Кол-во эпох
batch_size = 1000  # Размер пакета для мини-пакетного градиентного спуска
gamma = 0.8  # Гамма для алгоритма Нестерова
eps = 10**-6  # Эпсилон
len_hidden_layer = 64  # Кол-во нейронов в скрытом слое
num_layers = 3  # Общее кол-во слоёв в нейронной сети, включая входной и выходной

# Скачивание выборок
def download():
    dest_file_name = os.path.join(data_root, cifar_file_name)
    if not os.path.exists(dest_file_name):
        urlretrieve(url + cifar_file_name, dest_file_name)
    statinfo = os.stat(dest_file_name)
    print("File ", dest_file_name, " size : ", statinfo.st_size, " bytes.")
    return dest_file_name

# Извлечение выборок из скаченного архива
def extract(file_name):
    tar = tarfile.open(file_name)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
    dest_dir_name = os.path.join(data_root, cifar_dir_name)
    if not os.path.exists(dest_dir_name):
        raise Exception("No data directory. Check that archive with data exists, not corrupted and contains ",
                        cifar_dir_name, " inside.")
    return dest_dir_name

# Получение обучающей последовательности
def get_training_sets(data_dir_name):
    train_batches = ["data_batch_1",
                     "data_batch_2",
                     "data_batch_3",
                     "data_batch_4",
                     "data_batch_5",
                     ]
    train_labels = []
    train_data = []
    for train_batch in train_batches:
        path_to_batch = data_dir_name + "\\cifar-10-batches-py\\" + train_batch
        print("Loading ", path_to_batch, "...")
        batch = pickle.load(open(path_to_batch, "rb"), encoding="bytes")
        train_labels.extend(batch[b"labels"])
        train_data.extend(batch[b"data"])
        print("Done.")
    raw = pickle.load(open(data_dir_name + "\\cifar-10-batches-py\\batches.meta", "rb"))
    class_names = [x for x in raw["label_names"]]
    print(class_names)
    return train_data, train_labels, class_names

# Подготовка обучающих данных
def preprocess_data(X, m, v):
    if not m.any():
        m = np.mean(X, axis=0)
    if not v.any():
        v = np.var(X, axis=0)
    X = (X - m) / v
    return X, m, v

# Подготовка обучающех меток
def preprocess_labels(Y, num_classes):
    y_gt = np.zeros((len(Y), num_classes))
    for i in range(len(Y)):
        y_gt[i, Y[i]] = 1
    return y_gt

# (h2) Активационная функция
def activation(z):
    return 1.0 / (1.0 + np.exp(-z))

# Подсчёт h при прямом распространении ошибки
def h(theta, X):
    return 1.0 / (1.0 + np.exp(-np.dot(X, theta)))

# (h_prime) Обратная актвиационная функция для подсчёта Дельты в обратном распространении ошибки
def delta_activation(z):
    return activation(z) * (1 - activation(z))

# Потеря значений между получившимся и предсказанным значениями
def loss(y_gt, y_pred):
    cross_entropy_error = 0.0
    for i in range(0, len(y_gt)):
        for j in range(0, len(y_gt[i])):
            cross_entropy_error -= (y_gt[i][j] * np.log(y_pred[i][j]) + (1.0 - y_gt[i][j]) * np.log(1.0 - y_pred[i][j]))
    return cross_entropy_error / len(y_gt)

# Прямое распространение (Возможна ошибка. Если будет, то переделать, как у Вани)
def forward(theta_list, X):
    layer = X
    for l in range(0, len(theta_list)):
        layer = np.array([h(theta_list[l][i, :], layer) for i in range(0, theta_list[l].shape[0])])
        layer = np.transpose(layer)
    return layer

# Подсчёт производной от потерь кросс-энтропии по параметрам одного логистического классификатора
def derivative(X, y_gt, y_pred):
    return (y_pred - y_gt) * X

# Замена значений нейронов
def neuron_change(y_gt, y_pred):
    return derivative(y_gt, y_pred) * delta_activation(y_pred)

# Обратное распространение ошибки
def backward(X, y_gt, theta_list):
    num_features = X.shape[1]
    num_classes = y_gt.shape[1]
    d_theta1 = np.zeros([len_hidden_layer, num_features])
    d_theta2 = np.zeros([num_classes, len_hidden_layer])
    avg_delta_list = [d_theta1, d_theta2]
    for k in range(batch_size):
        zs = [np.zeros(len_hidden_layer, dtype=float), np.zeros(num_classes, dtype=float)]
        # Запись всех значений активации от уровня к уровню
        activations = [np.zeros(num_features, dtype=float), np.zeros(len_hidden_layer, dtype=float),
                       np.zeros(num_classes, dtype=float)]
        activations[0] = X[k]
        zs[0] = np.array([np.dot(theta_list[0][i, :], activations[0].T) for i in range(theta_list[0].shape[0])])
        activations[1] = activation(zs[0])
        for j in range(1, num_layers - 1):  # Цикл по скрытым слоям
            zs[j] = np.array([np.dot(theta_list[j][i, :], activations[j].T) for i in range(theta_list[j].shape[0])])
            activations[j + 1] = activation(zs[j])
        d_theta1 = np.zeros([len_hidden_layer, num_features])
        d_theta2 = np.zeros([num_classes, len_hidden_layer])
        d1 = np.zeros([len_hidden_layer])
        d2 = np.zeros([num_classes])
        d_list = [d1, d2]
        delta_list = [d_theta1, d_theta2]
        y_pred = activations[-1]
        for class_num in range(num_classes):
            d_list[-1][class_num] = y_gt[k, class_num] - y_pred[class_num]
        delta_list[-1] = np.outer(d_list[-1], activations[-2])
        for l in range(2, num_layers):
            sp = delta_activation(zs[-l])
            d_list[-l] = np.dot(theta_list[-l + 1].T, d_list[-l + 1]) * sp
            delta_list[-l] = np.outer(d_list[-l], activations[-l - 1])
        for i in range(len(delta_list)):
            avg_delta_list[i] = np.add(avg_delta_list[i], delta_list[i])
    for i in range(len(avg_delta_list)):
        avg_delta_list[i] = np.divide(avg_delta_list[i], batch_size)
    return avg_delta_list

# Основная часть программы
start_time = time.time()

file_name = download()
dir_name = extract(file_name)
train_data, train_labels, class_names = get_training_sets(dir_name)

num_classes = len(class_names)
train_data = np.array(train_data, dtype=float)
train_labels = np.array(train_labels)

print("Data shape:", train_data.shape)
print("Labels shape:", train_labels.shape)

num_features = train_data.shape[1]
num_samples = train_data.shape[0]
num_train_samples = 40000

p = np.random.permutation(num_samples)
x_train = train_data[p[0:num_train_samples], :]
y_train = train_labels[p[0:num_train_samples]]

x_val = train_data[p[num_train_samples:], :]
val_labels = train_labels[p[num_train_samples:]]
train_labels = y_train

x_train, m_train, v_train = preprocess_data(x_train, np.array([]), np.array([]))
y_train = preprocess_labels(y_train, num_classes)

x_val, _, _ = preprocess_data(x_val, m_train, v_train)

theta1 = np.random.randn(len_hidden_layer, num_features)
theta2 = np.random.randn(num_classes, len_hidden_layer)
theta_list = [theta1, theta2]  # Массив, который сохраняет значение тетта для того,
# чтобы можно было продолжить обучения сети, а не начинать его сначала
num_batches = num_train_samples // batch_size

y_pred = forward(theta_list, x_train)
loss_val = loss(y_train, y_pred)
print("Initial Loss:", loss_val)
loss_array = np.zeros([num_epochs])  # Массив для хранения значений потерь на всех эпохах.
accuracy_train_array = np.zeros([num_epochs])
accuracy_val_array = np.zeros([num_epochs])
vt = np.array([np.zeros([len_hidden_layer, num_features]), np.zeros([num_classes, len_hidden_layer])])
# Начало обучения
for i in range(num_epochs):
    p = np.random.permutation(num_train_samples)
    print('Epoch %d/%d ' % (i, num_epochs), end='.')
    for batch_num in range(num_batches):
        if np.mod(batch_num, num_batches // 10) == 0:
            print('.', end='')
            sys.stdout.flush()
        x_train_batch = x_train[p[batch_num * batch_size: (batch_num + 1) * batch_size], :]
        y_train_batch = y_train[p[batch_num * batch_size: (batch_num + 1) * batch_size]]
        # Дополнение для алгоритма Нестерова
        vt_gamma = np.array([np.multiply(vt[0], gamma), np.multiply(vt[1], gamma)])
        theta_list_tmp = [theta_list[n] - vt_gamma[n] for n in range(len(theta_list))]
        d_theta_list = backward(x_train_batch, y_train_batch, theta_list_tmp)

        for n in range(len(vt)):
            vt[n] = vt_gamma[n] + alpha * d_theta_list[n]
        for l in range(len(theta_list)):
            theta_list[l] = theta_list[l] + vt[l]

    pred_train_probs = forward(theta_list, x_train)
    pred_train_labels = np.argmax(pred_train_probs, axis=1)
    loss_val = loss(y_train, pred_train_probs)

    pred_val_labels = np.argmax(forward(theta_list, x_val), axis=1)
    accuracy_train = (1.0 - np.mean(pred_train_labels != train_labels)) * 100.0
    accuracy_val = (1.0 - np.mean(pred_val_labels != val_labels)) * 100.0
    print("\nLoss:", loss_val, "\nTrain accuracy:", accuracy_train, "\nValidation accuracy:", accuracy_val)
    loss_array[i] = loss_val
    accuracy_train_array[i] = accuracy_train
    accuracy_val_array[i] = accuracy_val

# Тестирование работы нейронной сети
test_samples = pickle.load(open(dir_name + "\\cifar-10-batches-py\\test_batch", "rb"), encoding="bytes")
test_labels = test_samples[b"labels"]
test_data = test_samples[b"data"]
x_test, _, _ = preprocess_data(test_data, m_train, v_train)
pred_test = np.argmax(forward(theta_list, x_test), axis=1)
print("\nTest accuracy:", 100 * np.mean(pred_test == test_labels))

# Сохранение результатов
np.savetxt('loss.txt', loss_array)
np.savetxt('train_accuracy.txt', accuracy_train_array)
np.savetxt('val_accuracy.txt', accuracy_val_array)
np.savetxt('pred.txt', pred_test)
np.save('last_theta.npy', theta_list)

finish_time = round(time.time(), 0)
minutes = round(finish_time / 60, 0)
hours = round(minutes / 60, 0)
print("Время работы программы:", hours, "час", minutes, "минуты", finish_time, "cекунды")