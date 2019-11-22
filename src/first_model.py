# import keras
import sys
import numpy as np
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from statistics import median

k = 3
seq_len = 500
comb = 4**k # possible combinations of k-mers

score_list = [float(line.rstrip('\n').split('\t')[1]) for line in open(sys.argv[2])]
med = median(score_list)
Y_data = np.asarray([[1,0] if score > med else [0,1] for score in score_list])
X_data = HDF5Matrix(sys.argv[1], 'traindata')
# X_data = fasta2np()
print(X_data.shape)
np.random.seed(2)


model = Sequential()
model.add(Convolution2D(150, (comb, 8), activation='relu', input_shape=( comb,seq_len-k, 1)))
# model.add(Convolution2D(1, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(X_data, Y_data, batch_size=1, epochs=10, verbose=1)

score = model.evaluate(X_data, Y_data, verbose=0)
print("Score is", score)
