from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow import keras
import numpy as np
import json

with open('tokenizer.json') as f:
    data = json.load(f)
    encoder = tokenizer_from_json(data)

with open('meta_data.json') as f:
    meta_data = json.load(f)

model = keras.models.load_model('trained_models/bi-lstm-model')

def predict_nationality(text):
    x_test = encoder.texts_to_sequences([text])
    max_length = meta_data.get("max_length")
    x_test = keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_length
    )
    x_test = np.array(x_test)
    y_pred = model.predict(x_test)[0]
    index = np.argmax(y_pred)
    return index, y_pred[index]

print(predict_nationality("Taverna"))