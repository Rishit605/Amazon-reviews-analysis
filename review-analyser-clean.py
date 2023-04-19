import matplotlib.pyplot as plt
import bz2
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential

data = '/kaggle/input/amazonreviews'

train = '/kaggle/input/amazonreviews/train.ft.txt.bz2'
test = '/kaggle/input/amazonreviews/test.ft.txt.bz2'


# DECOMPRESSING AND SAVING IT INTO A DATAFRAME

def get_labels_and_text(file):
    labels = []
    text = []

    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        text.append(x[10:].strip())

    labls = labels[:int(len(labels) * 0.01)]
    text = text[:int(len(labels) * 0.01)]

    return np.array(labels), text


# Calling the function to get Train and Test sets
train_labels, train_text = get_labels_and_text(train)
test_labels, test_text = get_labels_and_text(test)

train_df = pd.DataFrame(zip(train_text, train_labels), columns=['text', 'label'])
print(train_df.head())
test_df = pd.DataFrame(zip(test_text, test_labels), columns=['text', 'label'])
print(test_df.head())

# Checking the saved dataframe
print(train_df.head(10))

## SETTING UP PARAMETERS ##

vocab_size = 10000
embed_dim = 64
max_length = 120
trunc_type = 'pre'
oov_tok = "<OOV>"

## TOKENIZATION ##

token = Tokenizer(num_words=vocab_size, oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
token.fit_on_texts(train_df['text'].values)

word_idx = token.word_index

# Sequencing the encoded text

train_seq = token.texts_to_sequences(train_df['text'].values)
train_padded = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type)

test_seq = token.texts_to_sequences(test_df['text'].values)
test_padded = pad_sequences(test_seq, maxlen=max_length)

## MODELING

model = Sequential([
    layers.Embedding(vocab_size, embed_dim, input_length=max_length),
    layers.Conv1D(128, 6, activation='relu'),
    layers.GlobalMaxPool1D(),

    layers.Dense(48, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

# Generating a model summary
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the compiled model
y_train = pd.get_dummies(train_df['label']).values
y_test = pd.get_dummies(test_df['label']).values
# [ #train_df['label'].values, #test_df['label']. values))] For binary entropy loss function usage
history = model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test))


# Plotting the accuracy-loss function graph curve for better insight.
def plot_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


plot_loss_acc(history)
