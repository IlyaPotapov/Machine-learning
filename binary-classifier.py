# В данной работе необходимо обучить бинарный классификатор. Целевая переменная находится в последнем столбце.
# Замена значений. Выделить список string значений и перевести из в float

from sklearn import grid_search, linear_model, metrics, cross_validation, datasets
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf

# Работать будем с DataFrame. Нужно будет преобразовать файл в DataFrame

start_time = time.time()

# Подсчёт кол-во элементов каждого класса
def count_labels(pred):
    num_one = 0
    num_zero = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            num_zero += 1
        else:
            num_one += 1
    print("\nКол-во нулей:", num_zero, "\nКол-во единиц:", num_one)

# Создание train_data и train_labels (C dataframe работает плохо)
def create_data_and_labels(df, num_table):
    val1 = 0
    val2 = 0
    if df.shape[1] == 16:
        val = df['15'].value_counts()
        val1 = val.index[0]
        val2 = val.index[1]
    class_names = np.array([val1, val2])
    matrix = df.as_matrix()
    # Так как у теста нет целевых значений, нужно брать полный размер.
    if num_table == 0:  # Номер таблицы показывает данные, с которыми ведётся работа (0 - train, 1 - test)
        train_data = np.array([[0.0 for i in range(1, df.shape[1] - 1)] for j in range(df.shape[0])])
    else:
        train_data = np.array([[0.0 for i in range(1, df.shape[1])] for j in range(df.shape[0])])
    train_labels = np.array([0.0 for i in range(df.shape[0])])
    for i in range(df.shape[0]):
        if len(matrix[i]) == 16:
            train_labels[i - 1] = matrix[i][15]
        for j in range(1, len(train_data[i])):
            train_data[i - 1][j - 1] = matrix[i][j]
    return train_data, train_labels, class_names

# Обработка пропусков (Аналог Imputer) и преобразование из string в float
# (Берём слово и заменяем его на число. Например: private - самое часто повторяемое => 0.0)
def data_preparation(df, name_csv):
    # Сбор данных о частоте повторения значений в каждом столбце
    print("Работа над", name_csv, "начата")
    most_freq = pd.Series([str(0.0) for i in range(df.shape[1])])  # Наиболее часто повторяющие значения в каждом столбце
    for i in range(df.shape[1]):  # Проходим по столбцам
        print("Обрабатывается столбец:", i)
        val_change = df[df.columns[i]].value_counts()  # Серия частоповторяемых значений
        most_freq[most_freq.index[i]] = val_change.index[0]
        print("Производится замена пропусков.")
        for j in range(df.shape[0]):  # Перебор по строкам
            if df[df.columns[i]].loc[df.index[j]] == " ?":  # Замена пропусков
                df[df.columns[i]].loc[df.index[j]] = most_freq[most_freq.index[i]]  # Подставляем индекс
        val_index = df[df.columns[i]].value_counts()  # Серия частоповторяемых значений
        if df.dtypes[i] == "object":
            print("Производится перевод данных из object в int для дальнейшей работы.")
            indexes = pd.Series([val_index.index[v] for v in range(val_index.shape[0])])
            for j in range(df.shape[0]):
                for v in range(indexes.shape[0]):
                    if df[df.columns[i]].loc[df.index[j]] == indexes.loc[indexes.index[v]]:
                        df[df.columns[i]].loc[df.index[j]] = indexes.index[v]
                        break
    #print(df.dtypes)
    df.to_csv(name_csv)
    print("Преобразование", name_csv, "выполнено")

# Основной код

# Получение выборок
#train = pd.read_csv("train2.csv", sep=',')
#test = pd.read_csv("test2.csv", sep=',')
#data_preparation(train, "train.csv")
#data_preparation(test, "test.csv")

train = pd.read_csv("train.csv", sep=',')
test = pd.read_csv("test.csv", sep=',')

# Матричное преобразование работает быстрее (Гораздо)
train_data, train_labels, class_names = create_data_and_labels(train, 0)
test_data, _, _ = create_data_and_labels(test, 1)

np.random.seed(0)

#classifier = linear_model.SGDClassifier(random_state=0)  # Линейный классификатор с стахостическим градинетным спуском
#parameters_grid = {
#    'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss'],
#    'penalty': ['l1', 'l2'],
#    'n_iter': [5, 6, 7, 8, 9, 10],
#    'alpha': np.linspace(0.0001, 0.001, num=5)
#}
#cv = cross_validation.StratifiedShuffleSplit(train_labels, n_iter=10, test_size=0.2, random_state=0)
#grid_cv = grid_search.GridSearchCV(classifier, parameters_grid, scoring='accuracy', cv=cv)
#print(grid_cv.fit(train_data, train_labels))
model = Sequential()
model.add(Dense(len(train_data[0]), input_dim=len(train_data[0]), activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=2000)
score = model.evaluate(train_data, train_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
y_pred = model.predict_classes(test_data)
count_labels(y_pred)
np.savetxt('pred2.txt', y_pred)
print("Время работы программы:", round(time.time() - start_time, 2), "cекунды")
