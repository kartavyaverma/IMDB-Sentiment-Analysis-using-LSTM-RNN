import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# Suppress some tensorflow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_features = 90000 
maxlen = 500
batch_size = 64

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print("Build model...")
model = Sequential()
model.add(Embedding(max_features, 64))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print("Train...")
# We use a short patience and only a few epochs to make this reasonably quick
earlystopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2, 
          validation_data=(x_test, y_test),
          callbacks=[earlystopping])

print("Evaluate model...")
results = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {results[1] * 100:.2f}%")

print("Saving model to lstm_imdb.h5...")
model.save("lstm_imdb.h5")
print("Model saved successfully!")
