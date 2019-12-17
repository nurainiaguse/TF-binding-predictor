# import keras
import sys
import numpy as np
import os
from Bio import SeqIO
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D
from keras.layers import Convolution1D, MaxPooling1D, LocallyConnected1D
from keras.utils import np_utils
from statistics import median
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import keras.callbacks
from sklearn.metrics import precision_recall_curve

results_dir = "../results-320k-8w-32d/"

class BestLost(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

try:
	os.mkdir(results_dir+sys.argv[3])
except:
	pass

# os.mkdir("../results/"+sys.argv[3])

k = 3
seq_len = 500
comb = 4 # possible combinations of k-mers
fasta_sequences = SeqIO.parse(open(sys.argv[1]),'fasta')
def writeh5(seqs,filename,offset_values=None):
    seqsnp=np.zeros((len(seqs),4,seq_len))

    mydict={'A':np.asarray([1,0,0,0]),'G':np.asarray([0,1,0,0]),'C':np.asarray([0,0,1,0]),'T':np.asarray([0,0,0,1]),'N':np.asarray([0,0,0,0]),'H':np.asarray([0,0,0,0]),'a':np.asarray([1,0,0,0]),'g':np.asarray([0,1,0,0]),'c':np.asarray([0,0,1,0]),'t':np.asarray([0,0,0,1]),'n':np.asarray([0,0,0,0])}
    n=0
    offset_values = np.zeros(len(seqs))
    for seq in seqs:
    	for i in range(seq_len):
    		# seqsnp[n,:,i] = np.concatenate([mydict[seq[i]], mydict[seq[i]][::-1]], axis=0) # need both strands represented
    		seqsnp[n,:,i] = mydict[seq[i]]
    	n = n+1
    seqsnp=seqsnp[:n,:,:]
    # seqsnp = seqsnp.reshape(seqsnp.shape[0],4,seq_len,1)
    return seqsnp
seqs=[str(fasta.seq) for fasta in fasta_sequences]

'''

Data preprocessing
 - Half of data is positive, half is negative. Any sequence with score greater than median is considered positive.

'''
X_data = writeh5(seqs,sys.argv[1]+'.ref.h5')
score_list = [float(line.rstrip('\n').split('\t')[1]) for line in open(sys.argv[2])]
med = median(score_list) # if score is greater than median, we consider it as binding. otherwise, non-binding
# Y_data = np.asarray([1 if score > med else 0 for score in score_list])
Y_data_prob = np.asarray([score for score in score_list])
X_data, X_test, Y_data_prob, Y_test_prob = train_test_split(X_data, Y_data_prob, test_size=0.25, random_state=42)
Y_data = np.asarray([1 if score > med else 0 for score in Y_data_prob.tolist()])
Y_test = np.asarray([1 if score > med else 0 for score in Y_test_prob.tolist()])
print(X_data.shape)
np.random.seed(2)

'''

Model creation

'''
hyperparameters = [320, 8, 32] # number of kernels, cnn window size, dense layers
model = Sequential()
model.add(Convolution1D(hyperparameters[0], hyperparameters[1], activation='relu', input_shape=(4, seq_len), data_format="channels_first"))
model.add(MaxPooling1D(pool_size=4, strides=4,data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Convolution1D(hyperparameters[0], hyperparameters[1], activation='relu',data_format="channels_first"))
model.add(MaxPooling1D(pool_size=4, strides=4,data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Convolution1D(hyperparameters[0], hyperparameters[1], activation='relu',data_format="channels_first"))

# model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))
model.add(Flatten())
# model.add(LocallyConnected2D(1, (1, 22)))
# model.add(MaxPooling2D(pool_size=(1,40)))
# print("current model output shape 7", model.output_shape)
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))
model.add(Dense(hyperparameters[2], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

'''

Model training
 - saves the best model according to loss value

'''

checkpoint_filename = results_dir+sys.argv[3]+"/weights.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')
best_lost = BestLost()
callbacks_list = [checkpoint, best_lost]
history = model.fit(X_data, Y_data, batch_size=1, epochs=50, verbose=0, validation_split=0.25, callbacks=callbacks_list)

'''

Adding history plots
1. Accuracy history plot
2. Loss history plot
3. ROC plot
4. PR plot

ROC and PR plot uses the best weights found during training (least loss)

'''
# 1. Accuracy history plot
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(results_dir+sys.argv[3]+"/acc_plot.pdf")

# 2. Loss history plot
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(results_dir+sys.argv[3]+"/loss_plot.pdf")

'''

Adding statistical plots
1. ROC plot
2. PR plot

ROC and PR plot uses the best weights found during training (least loss)

'''

# 1. Load the best model that was saved earlier, and save it in a separate file
model.load_weights(checkpoint_filename)
model.save(results_dir+sys.argv[3]+"/best_model.hdf5")

# 2. Evaluate the model to obtain score (evaluation of loss function) and accuracy
score, acc = model.evaluate(X_test, Y_test, verbose=0)

# 3. Make prediction
y_pred = model.predict(X_test).ravel()

# 4. Plot ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
auc_val = auc(fpr, tpr)
plt.figure(3)
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_val))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(results_dir+sys.argv[3]+"/roc_plot.pdf")

# 5. Plot PR curve
lr_precision, lr_recall, _ = precision_recall_curve(Y_test,y_pred)
auprc = auc(lr_recall, lr_precision)
plt.figure(4)
plt.plot(lr_recall, lr_precision, marker='.', label='Keras (area = {:.3f})'.format(auprc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUPRC curve')
plt.legend(loc='best')
plt.savefig(results_dir+sys.argv[3]+"/auprc_plot.pdf")

'''

Write some summary onto file
1. Score
2. Accuracy
3. AUROC
4. AUPRC

'''

with open(results_dir+sys.argv[3]+"/summary.txt", "w") as f:
	f.write(str(score))
	f.write("\n")
	f.write(str(acc))
	f.write("\n")
	f.write(str(auc_val))
	f.write("\n")
	f.write(str(auprc))
	f.write("\n")
	model.summary(print_fn=lambda x: f.write(x + '\n'))
