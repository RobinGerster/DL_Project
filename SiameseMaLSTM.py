import numpy as np
import pandas as pd
import pickle
from keras.utils import pad_sequences
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
import keras.backend as K
from keras.callbacks import EarlyStopping
import tensorflow

model_num = 1 #DEFINE THIS PER RUN (1,2,3)

# Load data
numb_representation = pickle.load(open("data/model{}_train.pkl".format(model_num), "rb"))
df_dev = pd.read_csv("data/Train set processed.csv").dropna()

# Pad number representations
max_seq_len = 20
numb_representation_padded = []
for q1, q2 in numb_representation:
    q1_padded = pad_sequences([q1], maxlen=max_seq_len, padding='pre')[0]
    q2_padded = pad_sequences([q2], maxlen=max_seq_len, padding='pre')[0]
    numb_representation_padded.append([q1_padded, q2_padded])

# Load embedding matrix
embeddings_matrix = np.load("data/embeddings_matrix_model_{}.npy".format(model_num))

# Siamese LSTM model
input_shape = (max_seq_len,)
input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)
embedding_layer = Embedding(
    embeddings_matrix.shape[0],
    embeddings_matrix.shape[1],
    weights=[embeddings_matrix],
    input_length=max_seq_len,
    trainable=False,
)
lstm_layer = LSTM(50)
x1 = embedding_layer(input_1)
x1 = lstm_layer(x1)
x2 = embedding_layer(input_2)
x2 = lstm_layer(x2)
distance_layer = Lambda(
    lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
)
distance = distance_layer([x1, x2])
model = Model(inputs=[input_1, input_2], outputs=distance)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
num_epochs = 15
batch_size = 128


x1 = np.array([numb_representation_padded[idx][0] for idx in range(len(numb_representation_padded))])
x2 = np.array([numb_representation_padded[idx][1] for idx in range(len(numb_representation_padded))])

early_stopping = EarlyStopping(monitor='loss', patience=2)
model.fit(x=[x1,x2],y=df_dev["is_duplicate"].values,batch_size=batch_size,epochs=num_epochs,validation_split=0.15, callbacks=[early_stopping])
model.save("model{}".format(model_num))