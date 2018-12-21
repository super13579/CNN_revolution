import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,concatenate,Input, concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam

#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Define convolution with batchnromalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
  
#Define Inception structure
def Inception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = Conv2D(branch1[0],(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2D(branch2[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2D(branch2[1],(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2D(branch3[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2D(branch3[1],(1,1), padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2D(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(branchpool)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x
  
#Build InceptionV1 model
def InceptionV1(width, height, depth, classes):
    
    inpt = Input(shape=(width,height,depth))

    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    x = Inception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
    x = Inception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #14x14x480

    x = Inception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #7x7x832

    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
    x = Inception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024

    #Using AveragePooling replace flatten
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x =Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(classes,activation='softmax')(x)
    
    model=Model(input=inpt,output=x)
    
    return model

InceptionV1_model = InceptionV1(224,224,3,17)
InceptionV1_model.summary()

InceptionV1_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
History = InceptionV1_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

#Plot Loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()