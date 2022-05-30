import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from prep_data import *

stateList = list(np.load('state_list.npz', allow_pickle=True)['state_list'])

lmfcc_data, targets = prepare_data()

input_dim = lmfcc_data[0].shape[0]
output_dim = len(stateList)

model = keras.Sequential()
model.add(layers.Dense(256, input_dim=input_dim, activation="relu"))
#model.add()

print(model.summary())