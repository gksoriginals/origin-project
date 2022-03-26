from sklearn.utils import class_weight
from origin.models.bi_lstm.data import create_dataset
from origin.utils.class_weights import generate_class_weights
from origin.utils.lr_finder import LRFinder
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import sys

DATA_PATH = 'data/dataset.csv'

batch_size = 25

np.set_printoptions(threshold=sys.maxsize)

X_train, y_train, X_test, y_test, classes, n_classes = create_dataset(DATA_PATH)

class_weight = generate_class_weights(y_train)

encoder = keras.preprocessing.text.Tokenizer(
    char_level=True
)

encoder.fit_on_texts(X_train)

X_train = encoder.texts_to_sequences(X_train)
X_test = encoder.texts_to_sequences(X_test)

tokenizer_json = encoder.to_json()

max_length = max(map(len, X_train))

meta_data = {
    "max_length": max_length,
}

with open("meta_data.json", "w") as f:
    json.dump(meta_data, f)


with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

embedding_dim = 4
# print(x_train.shape[1], x_train.shape[2])
print(x_train.shape)
print(y_train.shape)

# Create the model

print(encoder.index_word)

model = keras.Sequential()

model.add(layers.Embedding(len(encoder.index_word) + 1, embedding_dim))

model.add(layers.LSTM(128, activation="tanh",
                      return_sequences=True, dropout=.4))

model.add(layers.LSTM(128, activation="tanh",
                      return_sequences=False, dropout=.4))

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dense(n_classes, activation="softmax"))

optimizer = keras.optimizers.Adam(lr=3e-4)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.build(input_shape=x_train.shape)
model.summary()
"""
lr_finder = LRFinder(model)
STEPS_PER_EPOCH = np.ceil(len(x_train) / batch_size)
lr_finder.find(x_train, y_train, start_lr=3e-4, end_lr=1, class_weight=class_weight, epochs=10,
               steps_per_epoch=STEPS_PER_EPOCH)
learning_rates = lr_finder.get_learning_rates()
losses = lr_finder.get_losses()
best_lr = lr_finder.get_best_lr(sma=20)
print(best_lr)
K.set_value(model.optimizer.lr, best_lr)
"""
earlystop_callback = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5)

history = model.fit(x=x_train, y=y_train, epochs=25, shuffle=True, class_weight=class_weight,
          batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[earlystop_callback])

model.save('trained_models/bi-lstm-model')
