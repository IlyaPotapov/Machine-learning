import scipy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import io

def preparing_data():
    x_train = []
    y_train = ()
    for i in range(7):
        file_name = str(i) + '_train.txt'
        with io.open(file_name, encoding='utf-8') as f:
            info = f.readlines()
            x_train = np.append(x_train, info)
            y_train = np.append(y_train, np.array(np.repeat(i, len(info))))
    return x_train, y_train

x_train, y_train = preparing_data()
count_vectirizer = CountVectorizer()  # Собираем ключевые слова. Считаем сколько раз они встретились в каждой строке
x_train = count_vectirizer.fit_transform(x_train)
tt = TfidfTransformer()  # Снижение веса слова. Отсеивание неинформативных слов.
x_train = tt.fit_transform(x_train)
with open('test.txt', encoding='utf-8') as f:
    x_test = f.readlines()
model = LogisticRegression()
model.fit(x_train, y_train)
model.predict(count_vectirizer.transform(x_test)).tofile("pred.txt", sep="\n")
