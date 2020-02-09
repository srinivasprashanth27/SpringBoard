import pandas as pd
import numpy as np
import os
import string
import fnmatch
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import cv2
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Convolution2D,MaxPooling2D
from keras.models import Model
from keras.layers.normalization import  BatchNormalization
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from time import time
import shutil
# tf.logging.set_verbosity(tf.logging.ERROR)
char_list = string.ascii_letters + string.digits



def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst




path = './mnt/ramdisk/max/90kDICT32px'


li_dirnames = []
li_filenames = []
# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []




max_label_len = 0

i = 1
flag = 0

for root, dirnames, filenames in tqdm(os.walk(path)):
    li_dirnames.append(dirnames)
    li_filenames.append(filenames)
    for f_name in fnmatch.filter(filenames, '*.jpg'):
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)

        # convert each image of shape (32, 128, 1)
        w, h = img.shape
        if h > 128 or w > 32:
            continue
        if w < 32:
            add_zeros = np.ones((32 - w, h)) * 255
            img = np.concatenate((img, add_zeros))

        if h < 128:
            add_zeros = np.ones((32, 128 - h)) * 255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis=2)

        # Normalize each image
        img = img / 255.

        # get the text from the image
        txt = f_name.split('_')[1]

        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)

        # split the 200000 data into validation and training dataset as 10% and 90% respectively
        if i % 10 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

            # break the loop if total data is 150000
        if(i%100==0):
            print('{} images processed'.format(i))

        if i == 200000:
            flag = 1
            break
        i += 1
    if flag == 1:
        break

# pad each output label to maximum text length



print("Storing the values to Array...")
print('Storing the data to local disk..')
np.save('training_img',training_img)
np.save('training_txt',training_txt)
np.save('train_input_length',train_input_length)
np.save('train_label_length',train_label_length)
np.save('orig_txt',orig_txt)
np.save('valid_img',valid_img)
np.save('valid_txt',valid_txt)
np.save('valid_input_length',valid_input_length)
np.save('valid_label_length',valid_label_length)
np.save('valid_orig_txt',valid_orig_txt)





print('Training Set Stats...')
print('Train image',len(training_img))
print('Train text',len(training_txt))
print('Validation data Stats...')
print('valid_img',len(valid_img))
print('Valid Text',len(valid_txt))
print('Max Label_length',max_label_len)


# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPooling2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
print(act_model.summary())





labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')




def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

print('Reading pickle files..')
training_img=np.load('training_img.npy',allow_pickle=True)
training_txt=np.load('training_txt.npy',allow_pickle=True)
train_input_length=np.load('train_input_length.npy',allow_pickle=True)
train_label_length=np.load('train_label_length.npy',allow_pickle=True)
orig_txt=np.load('orig_txt.npy',allow_pickle=True)
valid_img=np.load('valid_img.npy',allow_pickle=True)
valid_txt=np.load('valid_txt.npy',allow_pickle=True)
valid_input_length=np.load('valid_input_length.npy',allow_pickle=True)
valid_label_length=np.load('valid_label_length.npy',allow_pickle=True)
valid_orig_txt=np.load('valid_orig_txt.npy',allow_pickle=True)
max_label_len=22

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

tensorboard_dir=os.path.join("logs",str(time()))
pathh=os.path.join(os.getcwd(),"logs")
if os.path.exists(pathh):
    shutil.rmtree(pathh)
tensorboard=TensorBoard(log_dir=tensorboard_dir)

filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint,tensorboard]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 128
epochs = 50
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs, validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)

