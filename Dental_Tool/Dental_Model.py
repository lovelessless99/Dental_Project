from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.densenet import DenseNet121
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, load_model
from keras import models, layers, Model
from keras import backend as K
from keras.layers import *
import keras

def Simple_CNN_Net(input_shape, classes):
        model = models.Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        
#         model.add(Conv2D(128, (3, 3), activation='relu'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D((2, 2)))
       
        
#         model.add(Conv2D(256, (3, 3), activation='relu'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D((2, 2)))
        
        
        model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
        
#         model.add(Dense(64, activation='relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
        
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model
    
    
    
def CNN_Net(input_shape, classes):
        model = models.Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
       
        
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model
    

def Inception(input_shape, classes):
        
        input_tensor = Input(shape=input_shape)
        base_model = InceptionV3(input_tensor=input_tensor, include_top=False, weights=None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=predictions)
        
        
                
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])        
        return model
    
def ResNet(input_shape, classes):
        
        input_tensor = Input(shape=input_shape)
        base_model = ResNet50V2(input_tensor=input_tensor, include_top=False, weights=None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=input_tensor, outputs=predictions)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])        
        return model
    

def VGGNet(input_shape, classes):
        model = Sequential([
                Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Conv2D(128, (3, 3), activation='relu', padding='same',),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                
                Conv2D(256, (3, 3), activation='relu', padding='same',),
                Conv2D(256, (3, 3), activation='relu', padding='same',),
                Conv2D(256, (3, 3), activation='relu', padding='same',),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                Conv2D(512, (3, 3), activation='relu', padding='same',),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Flatten(),
                Dense(4096, activation='relu'),
                Dense(4096, activation='relu'),
                Dense(classes, activation='softmax')
        ])
        
        

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])        
        return model

def DenseNet(input_shape, classes):
            
        input_tensor = Input(shape=input_shape)
        base_model = DenseNet121(input_tensor=input_tensor, include_top=False, weights=None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=input_tensor, outputs=predictions)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])        
        return model