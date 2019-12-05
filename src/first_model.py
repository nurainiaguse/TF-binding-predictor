# import keras
import sys
import numpy as np
from Bio import SeqIO
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from statistics import median
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

k = 3
seq_len = 500
comb = 8 # possible combinations of k-mers
fasta_sequences = SeqIO.parse(open(sys.argv[1]),'fasta')
def writeh5(seqs,filename,offset_values=None):
    seqsnp=np.zeros((len(seqs),8,seq_len))

    mydict={'A':np.asarray([1,0,0,0]),'G':np.asarray([0,1,0,0]),'C':np.asarray([0,0,1,0]),'T':np.asarray([0,0,0,1]),'N':np.asarray([0,0,0,0]),'H':np.asarray([0,0,0,0]),'a':np.asarray([1,0,0,0]),'g':np.asarray([0,1,0,0]),'c':np.asarray([0,0,1,0]),'t':np.asarray([0,0,0,1]),'n':np.asarray([0,0,0,0])}
    n=0
    offset_values = np.zeros(len(seqs))
    for seq in seqs:
    	for i in range(seq_len):
    		seqsnp[n,:,i] = np.concatenate([mydict[seq[i]], mydict[seq[i]][::-1]], axis=0) # need both strands represented
    	n = n+1
    seqsnp=seqsnp[:n,:,:]
    seqsnp = seqsnp.reshape(seqsnp.shape[0],8,seq_len,1)
    return seqsnp
seqs=[str(fasta.seq) for fasta in fasta_sequences]

X_data = writeh5(seqs,sys.argv[1]+'.ref.h5')
score_list = [float(line.rstrip('\n').split('\t')[1]) for line in open(sys.argv[2])]
med = median(score_list) # if score is greater than median, we consider it as binding. otherwise, non-binding
Y_data = np.asarray([1 if score > med else 0 for score in score_list])
X_data, X_test, Y_data, Y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=42)
print(X_data.shape)
np.random.seed(2)


model = Sequential()
model.add(Convolution2D(150, (4, 8), strides=(4,1), activation='relu', input_shape=( comb,seq_len, 1)))
model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
model.add(Dropout(0.25))
model.add(Convolution2D(200, (1, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,4), strides=(1,4)))
model.add(Dropout(0.25))
model.add(Convolution2D(500, (1, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Flatten())
# model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(printing)


history = model.fit(X_data, Y_data, batch_size=1, epochs=50, verbose=0, validation_split=0.25)

# summarize history for accuracy (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../results/"+sys.argv[3]+".pdf")

score, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Test score:", score) # evaluation of the loss function for a given input
print("Test accuracy:", acc)

with open("../results/"+sys.argv[3]+".txt", "w") as f:
	f.write(str(score))
	f.write("\n")
	f.write(str(acc))
