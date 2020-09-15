from keras.models import Sequential, load_model
from keras import models, layers, Model
from keras import backend as K
from keras.layers import *
import keras

def Inception_Module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
        
        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
        
        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
        
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
        
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out
    
    
def Residual_Module(layer_in, n_filters):
        merge_input = layer_in
        
        # check if the number of filters needs to be increase, assumes channels last format
        if layer_in.shape[-1] != n_filters:
            merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        
        # conv1
        conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        
        # conv2
        conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
       
        # add filters, assumes filters/channels last
        layer_out = add([conv2, merge_input])
        
        # activation function
        layer_out = Activation('relu')(layer_out)
        return layer_out  
    

def conv_block(x, nb_filter, kernel_size, padding='same', strides=(1, 1), use_bias=False):
        channel_axis = -1
        x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x
    
def inception_stem(inputs):
        channel_axis = -1
        # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
        x = conv_block(inputs, 32, (3, 3), strides=(2, 2), padding='valid')
        x = conv_block(x, 32, (3, 3), padding='valid')
        x = conv_block(x, 64, (3, 3))

        x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
        x2 = conv_block(x, 96, (3, 3), strides=(2, 2), padding='valid')
        x = concatenate([x1, x2], axis=channel_axis)
        #x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

        x1 = conv_block(x, 64, (1, 1))
        x1 = conv_block(x1, 96, (3, 3), padding='valid')

        x2 = conv_block(x, 64,  (1, 1))
        x2 = conv_block(x2, 64, (1, 7))
        x2 = conv_block(x2, 64, (7, 1))
        x2 = conv_block(x2, 96, (3, 3), padding='valid')

        #x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
        x = concatenate([x1, x2], axis=channel_axis)

        x1 = conv_block(x, 192, (3, 3), strides=(2, 2), padding='valid')
        x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

        #x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
        x = concatenate([x1, x2], axis=channel_axis)

        return x


def inception_A(input):
        channel_axis = -1

        a1 = conv_block(input, 96, (1, 1))

        a2 = conv_block(input, 64, (1, 1))
        a2 = conv_block(a2, 96, (3, 3))

        a3 = conv_block(input, 64, (1, 1))
        a3 = conv_block(a3, 96, (3, 3))
        a3 = conv_block(a3, 96, (3, 3))

        a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        a4 = conv_block(a4, 96, (1, 1))

       # m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
        m = concatenate([a1, a2, a3, a4], axis=channel_axis)

        return m


def inception_B(inputs):
        channel_axis = -1

        b1 = conv_block(inputs, 384, (1, 1))

        b2 = conv_block(inputs, 192, (1, 1))
        b2 = conv_block(b2, 224, (1, 7))
        b2 = conv_block(b2, 256, (7, 1))

        b3 = conv_block(inputs, 192, (1, 1))
        b3 = conv_block(b3, 192, (7, 1))
        b3 = conv_block(b3, 224, (1, 7))
        b3 = conv_block(b3, 224, (7, 1))
        b3 = conv_block(b3, 256, (1, 7))

        b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        b4 = conv_block(b4, 128, (1, 1))

        #m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
        m = concatenate([b1, b2, b3, b4], axis=channel_axis)

        return m


def inception_C(inputs):
        channel_axis = -1

        c1 = conv_block(inputs, 256, (1, 1))

        c2 = conv_block(inputs, 384, (1, 1))
        c2_1 = conv_block(c2, 256, (1, 3))
        c2_2 = conv_block(c2, 256, (3, 1))
       # c2 = merge([c2_1, c2_2], mode='concat', concat_axis=channel_axis)
        c2 = concatenate([c2_1, c2_2], axis=channel_axis)


        c3 = conv_block(inputs, 384, (1, 1))
        c3 = conv_block(c3, 448, (3, 1))
        c3 = conv_block(c3, 512, (1, 3))
        c3_1 = conv_block(c3, 256, (1, 3))
        c3_2 = conv_block(c3, 256, (3, 1))
        #c3 = merge([c3_1, c3_2], mode='concat', concat_axis=channel_axis)
        c3 = concatenate([c3_1, c3_2], axis=channel_axis)

        c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        c4 = conv_block(c4, 256, (1, 1))

       # m = merge([c1, c2, c3, c4], mode='concat', concat_axis=channel_axis)
        m = concatenate([c1, c2, c3, c4], axis=channel_axis)

        return m


def reduction_A(inputs):
        channel_axis = -1

        r1 = conv_block(inputs, 384, (3, 3), strides=(2, 2), padding='valid')

        r2 = conv_block(inputs, 192, (1, 1))
        r2 = conv_block(r2, 224, (3, 3))
        r2 = conv_block(r2, 256, (3, 3), strides=(2, 2), padding='valid')

        r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

        #m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
        m = concatenate([r1, r2, r3], axis=channel_axis)

        return m


def reduction_B(inputs):
        channel_axis = -1

        r1 = conv_block(inputs, 192, (1, 1))
        r1 = conv_block(r1, 192, (3, 3), strides=(2, 2), padding='valid')

        r2 = conv_block(inputs, 256, (1, 1))
        r2 = conv_block(r2, 256, (1, 7))
        r2 = conv_block(r2, 320, (7, 1))
        r2 = conv_block(r2, 320, (3, 3), strides=(2, 2), padding='valid')

        r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

        #m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
        m = concatenate([r1, r2, r3], axis=channel_axis)

        return m

def create_inception_v4(input_shape, classes):
        '''
        Creates a inception v4 network
        :param nb_classes: number of classes.txt
        :return: Keras Model with 1 input and 1 output
        '''

        init = Input(shape=input_shape)

        x = inception_stem(init)

        # 4 x Inception A
        for i in range(4): x = inception_A(x)
#         for i in range(2): x = inception_A(x)
        
        
        # Reduction A
        x = reduction_A(x)

        # 7 x Inception B
        for i in range(7): x = inception_B(x)
#         for i in range(2): x = inception_B(x)


        # Reduction B
        x = reduction_B(x)

        # 3 x Inception C
        for i in range(3): x = inception_C(x)
#         for i in range(1): x = inception_C(x)
        

        # Average Pooling
        x = AveragePooling2D((4, 4))(x)

        # Dropout
        x = Dropout(0.2)(x)
        x = Flatten()(x)

        # Output
        out = Dense(classes, activation='softmax')(x)

        model = Model(init, out, name='Inception-v4')
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True), metrics=["accuracy"])
        return model