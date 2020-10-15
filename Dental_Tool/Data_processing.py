from keras.utils import to_categorical, plot_model
from tqdm.notebook import tqdm as tqdm
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

import collections
import random
import keras
import glob
import json
import cv2
import sys
import os



def split_dataframe(dataframe, arguscale=20):
            data_volume = len(dataframe) // arguscale
            training_range = (0, int(round(data_volume * 0.6, 0)))
            valid_range    = (training_range[1], int(training_range[1] + round(data_volume * 0.2, 0)) )
            test_range     = (valid_range[1], int(valid_range[1] + round(data_volume * 0.2, 0)))


            training_idx_range     = ( training_range[0] * arguscale, training_range[1] * arguscale )
            valid_idx_range        = ( valid_range[0] * arguscale   , valid_range[1]    * arguscale )
            test_idx_range         = ( test_range[0] * arguscale    , test_range[1]     * arguscale )

            train_data = dataframe.iloc[training_idx_range[0]:training_idx_range[1], :].reset_index(drop=True)
            valid_data = dataframe.iloc[valid_idx_range[0]:valid_idx_range[1], :].reset_index(drop=True)
            test_data  = dataframe.iloc[test_idx_range[0]:, :].reset_index(drop=True)

            train_data = shuffle(train_data).reset_index(drop=True)
            valid_data = shuffle(valid_data).reset_index(drop=True)
            test_data  = shuffle(test_data).reset_index(drop=True)

            return train_data, valid_data, test_data  


class DataGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs, labels, batch_size=32, dim=(256, 256), n_channels=1,
                     n_classes=10, shuffle=True, resize_setting=(256, 256)):
                'Initialization'
                self.dim = dim
                self.batch_size = batch_size
                self.labels = labels
                self.list_IDs = list(list_IDs)
                self.n_channels = n_channels
                self.n_classes = n_classes
                self.shuffle = shuffle
                self.on_epoch_end()
                self.resize_setting = resize_setting
                

        def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.ceil(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
                'Generate one batch of data'
                # Generate indexes of the batch
                end = (index+1)*self.batch_size if (index+1)*self.batch_size < len(self.indexes) else len(self.indexes)
                indexes = self.indexes[index*self.batch_size:end] 

                # Find list of IDs
                list_IDs_temp = [self.list_IDs[k] for k in indexes]         
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
               
                return X, y

        def on_epoch_end(self):
                'Updates indexes after each epoch'
                self.indexes = np.arange(len(self.list_IDs))
                if self.shuffle == True:
                        np.random.shuffle(self.indexes)
        
        def noisy_image(self, noise_typ, image):
                if noise_typ == "gauss":
                          row,col= image.shape
                          mean = 0
                          var = 0.1
                          sigma = var**0.5
                          gauss = np.random.normal(mean,sigma,(row,col))
                          gauss = gauss.reshape(row,col)
                          noisy = image + gauss
                          return noisy

                elif noise_typ == "s&p":
                          row,col = image.shape
                          s_vs_p = 0.5
                          amount = 0.004
                          out = np.copy(image)
                          # Salt mode
                          num_salt = np.ceil(amount * image.size * s_vs_p)
                          coords = [np.random.randint(0, i - 1, int(num_salt))
                                  for i in image.shape]
                          out[coords] = 1

                          # Pepper mode
                          num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                          coords = [np.random.randint(0, i - 1, int(num_pepper))
                                  for i in image.shape]
                          out[coords] = 0
                          return out

                elif noise_typ == "poisson":
                          vals = len(np.unique(image))
                          vals = 2 ** np.ceil(np.log2(vals))
                          noisy = np.random.poisson(image * vals) / float(vals)
                          return noisy

                elif noise_typ == "speckle":
                          row,col = image.shape
                          gauss = np.random.randn(row,col)
                          gauss = gauss.reshape(row,col)        
                          noisy = image + image * gauss
                          return noisy
            
        def __data_generation(self, list_IDs_temp):
            
                # Initialization
                image_array_size = ( self.batch_size, *self.dim, self.n_channels )
                X = np.zeros(image_array_size, dtype=np.uint8)
                y = np.array([0]* self.batch_size)
                
                for i, ID in enumerate(list_IDs_temp):
                        image = cv2.imread(ID, 0)
                        image = cv2.resize(image, (self.resize_setting[1], self.resize_setting[0]))
                        
#                         from random import sample
#                         random_index = sample(range(0, 32), k=8)
                
#                         if i in random_index: image = self.noisy_image("gauss", image)
                        image = np.reshape(image, (*image.shape, 1))
                        X[i,], y[i] =  image, self.labels[ID]
                       
                        
                return X, to_categorical(y, num_classes=self.n_classes)
            


def get_generator(dataframe, argscale=20, batch_size=32):
        
        stage_0 = dataframe[dataframe["State"] == 0]
        stage_1 = dataframe[dataframe["State"] == 1]
        stage_2 = dataframe[dataframe["State"] == 2]
        stage_3 = dataframe[dataframe["State"] == 3]
        
        train_data_0, valid_data_0, test_data_0 = split_dataframe(stage_0.reset_index(drop=True), arguscale=argscale) 
        train_data_1, valid_data_1, test_data_1 = split_dataframe(stage_1.reset_index(drop=True), arguscale=argscale) 
        train_data_2, valid_data_2, test_data_2 = split_dataframe(stage_2.reset_index(drop=True), arguscale=argscale) 
        train_data_3, valid_data_3, test_data_3 = split_dataframe(stage_3.reset_index(drop=True), arguscale=argscale) 
        
        train_min, valid_min, test_min = len(train_data_3), len(valid_data_3), len(test_data_3)
#         train_min, valid_min, test_min = len(train_data_2), len(valid_data_2), len(test_data_2)

        train_dataset   = pd.concat([train_data_0.iloc[:train_min//2, :], train_data_1.iloc[:train_min//2, :], 
                                     train_data_2.iloc[:train_min, :], train_data_3])
        
        valid_dataset   = pd.concat([valid_data_0.iloc[:valid_min//2, :], valid_data_1.iloc[:valid_min//2, :], 
                                     valid_data_2.iloc[:valid_min, :], valid_data_3])
        
        test_dataset    = pd.concat([test_data_0.iloc[:test_min//2, :] , test_data_1.iloc[:test_min//2, :] , 
                                     test_data_2.iloc[:test_min, :] , test_data_3 ])
                                         
        
        train_dataset   = shuffle(train_dataset)
        train_generator = make_generator(train_dataset, batch_size)
        
        valid_dataset   = shuffle(valid_dataset)
        valid_generator = make_generator(valid_dataset, batch_size)
        
        test_dataset    = shuffle(test_dataset)
        test_generator  = make_generator(test_dataset, batch_size)
        
        return train_dataset, valid_dataset, test_dataset, train_generator, valid_generator, test_generator

    
def balance_data_generator(dataframe, argscale=20, batch_size=32):
        train_dataset, valid_dataset, test_dataset = balance_class_and_tooth(dataframe, argscale)
       
        train_dataset   = shuffle(train_dataset)
        train_generator = make_generator(train_dataset, batch_size)
        
        valid_dataset   = shuffle(valid_dataset)
        valid_generator = make_generator(valid_dataset, batch_size)
        
        test_dataset    = shuffle(test_dataset)
        test_generator  = make_generator(test_dataset, batch_size)
       
        return train_dataset, valid_dataset, test_dataset, train_generator, valid_generator, test_generator
    

def make_generator(dataset, batch_size=32):
        common_params = {
                              'batch_size': batch_size,
                              'n_classes' : len(np.unique(dataset.Class)),
                              'n_channels': 1,
                              'shuffle'   : False,
                              'resize_setting': (200, 180),
                              'dim': (200, 180)
        }
        
        dataset_dict = collections.OrderedDict(zip(dataset.Path, dataset.Class))
        dataset_generator = DataGenerator(dataset["Path"], dataset_dict ,**common_params)
        return dataset_generator
        
def statisatic_dental_for_class(dataset):        
        dental_distribution = pd.DataFrame(index=[i for i in range(1, 33)])

        for i in range(4):
                data_frame = dataset[dataset["State"] == i]
                num = list(data_frame["tooth_num"])
                items, freqs = np.unique(num, return_counts=True)
                for item, freq in zip(items, freqs):
                        dental_distribution.loc[item,"Stage_%d"%i] = freq
        dental_distribution = dental_distribution.fillna(0)
        return dental_distribution
    
    
def print_class_ratio(dataframe):
        stage_0 = len(dataframe[dataframe["State"] == 0])
        stage_1 = len(dataframe[dataframe["State"] == 1])
        stage_2 = len(dataframe[dataframe["State"] == 2])
        stage_3 = len(dataframe[dataframe["State"] == 3])
        print("Class 0 : %d, Class 1 : %d, Class 2 : %d" % ( (stage_0 + stage_1), stage_2, stage_3 ))
        print("Stage 0 : %d, Stage 1 : %d, Stage 2 : %d, Stage 3 : %d" % ( stage_0, stage_1, stage_2, stage_3 ))
        
def get_ratio(real_tooth_amount):
        reminder  = real_tooth_amount
        train_end = round(real_tooth_amount * 0.6, 0)
        
        reminder -= train_end
        valid_end = reminder // 2
        
        reminder -= valid_end
        test_end = reminder
        
        return train_end, valid_end, test_end
    
def balance_teeth_distribution(dataset, argscale=20):
        train_dataset, valid_dataset, testing_dataset = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for stage in range(4):
                stage_dataframe = dataset[dataset["State"] == stage]
                tooth_group = stage_dataframe.groupby("tooth_num")
                for tooth_num, tooth_table in tooth_group:
                        tooth_table = tooth_table.reset_index(drop=True)
                        real_tooth_amount = len(tooth_table) // argscale
                        train_end, valid_end, test_end = 0, 0, 0

                        if real_tooth_amount == 0 : continue

                        if   real_tooth_amount == 1   : train_end, valid_end, test_end = 1, 0, 0
                        elif real_tooth_amount == 2   : train_end, valid_end, test_end = 1, 0, 1
                        elif real_tooth_amount == 3   : train_end, valid_end, test_end = 1, 1, 1
                        elif real_tooth_amount == 4   : train_end, valid_end, test_end = 2, 1, 1
                        else: train_end, valid_end, test_end = get_ratio(real_tooth_amount)
                        
                        train_range = (0, int(train_end * argscale)) 
                        valid_range = (train_range[1], int(train_range[1] + valid_end * argscale))
                        test_range  = (valid_range[1], int(valid_range[1] + test_end  * argscale))

#                         print("tooth_info : %s " % tooth_num)
#                         print("real_number: %s " % real_tooth_amount)

#                         print("train_teeth: %d, valid_teeth: %d, test_teeth: %d" % (train_end, valid_end, test_end))
#                         print("train_range: (%s, %s) " % (train_range[0], train_range[1]))
#                         print("valid_range: (%s, %s) " % (valid_range[0], valid_range[1]))
#                         print("test_range : (%s, %s) \n\n" % (test_range[0] , test_range[1] )       )

                        train_dataset   = pd.concat([train_dataset  , tooth_table[train_range[0]: train_range[1] ]], ignore_index=False)
                        valid_dataset   = pd.concat([valid_dataset  , tooth_table[valid_range[0]: valid_range[1] ]], ignore_index=False)
                        testing_dataset = pd.concat([testing_dataset, tooth_table[test_range[0] : test_range[1]  ]], ignore_index=False)
        return train_dataset, valid_dataset, testing_dataset
    
    
    
def adjust_class_ratio(dataframe, argscale):
        new_dataset = pd.DataFrame()
        stage_0 = len(dataframe[dataframe["State"] == 0])
        stage_1 = len(dataframe[dataframe["State"] == 1])
        stage_2 = len(dataframe[dataframe["State"] == 2])
        stage_3 = len(dataframe[dataframe["State"] == 3])
        
        print("stage 0: %d, stage 1: %d, stage 2: %d, stage 3: %d" %(stage_0, stage_1, stage_2, stage_3) )
        min_num = min(stage_0, stage_1, stage_2, stage_3)
        
        Class_nums = [ min_num//2, min_num//2, min_num, min_num ]
        Stages     = [ 0, 1, 2, 3]
        
        for Stage, Class_num in zip(Stages, Class_nums):
                stage_dataset = dataframe[dataframe["State"] == Stage].reset_index(drop=True)
                groups = [ stage_dataset.iloc[ i:i+argscale ,:] for i in range(0, len(stage_dataset), argscale) ]
                random.shuffle(groups)
                stage_dataset_shuffle = pd.concat(groups).reset_index(drop=True)
                get_enough_data_flag, count = False, 0

                sample_dict = collections.OrderedDict()
                appear_dict = {}
                while not get_enough_data_flag:        
                        for i in range(0, len(stage_dataset), argscale):
                                same_images = stage_dataset_shuffle.iloc[ i:i+argscale ,:].reset_index(drop=True)
                                if i not in appear_dict : appear_dict[i] = set()
                                
                                while True:
                                        random_idx = random.randint(0, argscale-1) 
                                        if random_idx not in appear_dict[i]: break
                                
                                appear_dict[i].add(random_idx)
                                append_data = same_images.iloc[random_idx, :]
                                sample_dict[count] = append_data.to_dict()
                                count += 1
                                if count >= Class_num:
                                        get_enough_data_flag = True
                                        break
                stage_sample_dataframe = pd.DataFrame().from_dict(sample_dict).T
                new_dataset = pd.concat([new_dataset, stage_sample_dataframe])             
        return new_dataset
    
    
def balance_class_and_tooth(dataframe, argscale):
        train_bal_tooth, valid_bal_tooth, test_bal_tooth = balance_teeth_distribution(dataframe, argscale)
        training_dataset = adjust_class_ratio(train_bal_tooth, argscale=argscale)
        valid_dataset    = adjust_class_ratio(valid_bal_tooth, argscale=argscale)
        testing_dataset  = adjust_class_ratio(test_bal_tooth , argscale=argscale)
        return training_dataset, valid_dataset, testing_dataset
    
    
    
