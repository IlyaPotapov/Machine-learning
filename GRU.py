# Работа с алгоритмом GRU
import io
import sys
import numpy as np
import keras
import tensorflow


def read_file(file_name, reading_strings='all'):
    # file_name - имя файла
    # reading_strings - кол-во строк, которые необходимо прочитать из файла.
    text = list()
    with io.open(file_name, encoding='utf-8') as f:
        if reading_strings == 'all':
            text = f.readlines()
        else:
            rs = 0
            for line in f:
                text.append(line)
                rs += 1
                if rs == int(reading_strings):
                    break
    return text

#text = read_file("Война и мир.txt")
text = read_file("Война и мир.txt", '1000')
#text = read_file("Война и мир.txt", '5000')
#text = read_file("Война и мир.txt", '10000')
#text = read_file("Война и мир.txt", '15000')
#text = read_file("Война и мир.txt", '20000')

print("Данные для обучения получены")

# Придумать классы, в которых алгоритм сможет работать.

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# build the model: a single LSTM
print('Build model...')
model = keras.models.Sequential()
model.add(keras.layers.GRU(128, input_shape=(maxlen, len(chars))))
model.add(keras.layers.Dense(len(chars)))
model.add(keras.layers.Activation('softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    #print()
    print('----- Generating text after Epoch: %d' % (epoch + 1))
    start_index = np.random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        #print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += str(sentence)
        #print('----- Generating with seed: "' + str(sentence) + '"')
        #sys.stdout.write(generated)
        keys = char_indices.keys()
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            if len(sentence) > maxlen:
                sentence = sentence[0:maxlen]
            for t, char in enumerate(sentence):
                key_item = 0
                for key in keys:
                    if char == key:
                        break
                    if key_item == len(keys) - 1:
                        char = '\n'
                    key_item += 1
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + list(next_char)
        #    sys.stdout.write(next_char)
        #    sys.stdout.flush()
        #print("Цикл", i)
    f = open('GRU_text_20000_epoch_' + str(epoch + 1) + '.txt', 'w')
    for index in generated:
        f.write(index)
    f.close()


print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=10, callbacks=[print_callback])