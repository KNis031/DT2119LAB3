# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
import h5py

# stateList = list(np.load('state_list.npz', allow_pickle=True)['state_list'])
# output_dim = len(stateList)


# model = keras.Sequential()
# model.add(layers.Dense(256, activation="relu", name='layer1'))
# model.add(layers.Dense(output_dim, activation='softmax', name='outputlayer'))

# #model.compile(optimizer=keras.optimizers('Adam'), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
# opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# y_train = tf.ones((256))
# x_train = tf.random.uniform((256, 91))
# y_test = tf.ones((25))
# x_test = tf.random.uniform((25, 91))

# model.fit(x_train, y_train, epochs=10, batch_size=int(256/4), verbose=1)
# print(model.summary())

# predictions = model.predict_classes(x_test)
# for i in range(5):
# 	print('%s => %d (expected %d)' % (x_test[i].tolist(), predictions[i], y_test[i]))

# a=np.load('mspec_and_targets.npz', allow_pickle=True)['data']
# print(np.shape(a))
# f2 = h5py.File('mspec_and_targets.hdf5', 'r')
# dset1 = f2['train_data']
# print(dset1.shape)

# stateList = list(np.load('state_list.npz', allow_pickle=True)['state_list'])
# print(stateList)
# state_to_phone = [state[:-2] for state in stateList] #t.ex för index 33,34,35 (ox_0, ox_1, ox_2): state_to_phone= [...ox,ox,ox...] 
# print(state_to_phone)

test_x = np.load('testdata.npz', allow_pickle=True)['testdata']
test_y = np.load('testdata.npz', allow_pickle=True)['testdata']
print(np.shape(test_x))

