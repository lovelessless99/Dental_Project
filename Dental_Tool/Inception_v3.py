from keras.models import Sequential, load_model
from keras import models, layers, Model
from keras import backend as K
from keras.layers import *
import keras


def conv2d_bn(x, nb_filter, num_row, num_col, conv_name=None, padding='same', strides=(1, 1), use_bias=False):
        channel_axis = -1
        
        x = Conv2D( nb_filter, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

def Simple_Inception(input_shape, classes):
        channel_axis = -1 
        input_tensor = Input(shape=input_shape)
        x = conv2d_bn(input_tensor, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
#         x = conv2d_bn(x, 64, 3, 3)
#         x = MaxPooling2D((3, 3), strides=(2, 2))(x)

#         x = conv2d_bn(x, 80, 1, 1, padding='valid')
#         x = conv2d_bn(x, 192, 3, 3, padding='valid')
#         x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(x, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(x, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        
        branch7x7dbl = conv2d_bn(x, 96, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 96, 7, 1)
        
         
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')
        
        
        
        
         # mixed 1: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')
        
        
#         x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Flatten()(x)
#         x = Dense(100)(x)
        predictions = Dense(classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=input_tensor, outputs=predictions)
        model.compile(loss=keras.losses.categorical_crossentropy,
#                       optimizer='rmsprop',
                      optimizer=keras.optimizers.Adadelta(lr=0.001),
                      metrics=['accuracy'])    
        return model

def Custom_Inception_V3(input_shape, classes):
        channel_axis = -1 
        input_tensor = Input(shape=input_shape)
        
        x = Cropping2D(cropping=((20, 50), (0, 0)))(input_tensor) # from top, bottom, left, right (200*180)
        
        x = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
        x = conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, )
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        
        # mixed 1: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        
        # mixed 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        
#         # mixed 3: 17 x 17 x 768
#         branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

#         branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#         branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#         branch3x3dbl = conv2d_bn(
#             branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

#         branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
#         x = layers.concatenate(
#             [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        
#         # mixed 4: 17 x 17 x 768
#         branch1x1 = conv2d_bn(x, 192, 1, 1)

#         branch7x7 = conv2d_bn(x, 128, 1, 1)
#         branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
#         branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#         branch7x7dbl = conv2d_bn(x, 128, 1, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#         branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#             axis=channel_axis,
#             name='mixed4')

        
#         # mixed 5, 6: 17 x 17 x 768
#         for i in range(2):
#                 branch1x1 = conv2d_bn(x, 192, 1, 1)

#                 branch7x7 = conv2d_bn(x, 160, 1, 1)
#                 branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
#                 branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#                 branch7x7dbl = conv2d_bn(x, 160, 1, 1)
#                 branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#                 branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
#                 branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#                 branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#                 branch_pool = AveragePooling2D(
#                     (3, 3), strides=(1, 1), padding='same')(x)
#                 branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#                 x = layers.concatenate(
#                     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#                     axis=channel_axis,
#                     name='mixed' + str(5 + i))

#         # mixed 7: 17 x 17 x 768
#         branch1x1 = conv2d_bn(x, 192, 1, 1)

#         branch7x7 = conv2d_bn(x, 192, 1, 1)
#         branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
#         branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#         branch7x7dbl = conv2d_bn(x, 192, 1, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#         branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#             axis=channel_axis,
#             name='mixed7')
        
       
                   
#         # mixed 8: 8 x 8 x 1280
#         branch3x3 = conv2d_bn(x, 192, 1, 1)
#         branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
#                               strides=(2, 2), padding='valid')

#         branch7x7x3 = conv2d_bn(x, 192, 1, 1)
#         branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
#         branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
#         branch7x7x3 = conv2d_bn(
#             branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

#         branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
#         x = layers.concatenate(
#             [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

#         # mixed 9: 8 x 8 x 2048
#         for i in range(2):
#                 branch1x1 = conv2d_bn(x, 320, 1, 1)

#                 branch3x3 = conv2d_bn(x, 384, 1, 1)
#                 branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
#                 branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
#                 branch3x3 = layers.concatenate(
#                     [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

#                 branch3x3dbl = conv2d_bn(x, 448, 1, 1)
#                 branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
#                 branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
#                 branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
#                 branch3x3dbl = layers.concatenate(
#                     [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

#                 branch_pool = AveragePooling2D(
#                     (3, 3), strides=(1, 1), padding='same')(x)
#                 branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#                 x = layers.concatenate(
#                     [branch1x1, branch3x3, branch3x3dbl, branch_pool],
#                     axis=channel_axis,
#                     name='mixed' + str(9 + i))
                
        
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = Dense(classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=input_tensor, outputs=predictions)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
#                       optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])    
        
        return model
        
        