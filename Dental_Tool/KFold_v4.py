from Dental_Tool.Data_processing import make_generator
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import collections
import random
import glob
import json
import cv2
import sys
import os


def split_K_Fold(dataframe, augscale, fold_num):
        total_stage_3 = len(dataframe[dataframe.State==3])
        
        def get_ID_frequence(dataframe, augscale):
                groups = [ table for patient_ID, table in dataframe.groupby("ID") ]
                ID_groups = dataframe.groupby("ID")
                frequence = []
                total_stage_3 = len(dataframe[dataframe.State==3])
                for group_ID, group_table in ID_groups:
                        frequence.append([group_ID, len(group_table[group_table.State==3]) // augscale])
                return frequence
        
        frequence = get_ID_frequence(dataframe, augscale)
        fraction = round( total_stage_3 / augscale / fold_num )
        np.random.shuffle(frequence)
        
        fold_index = [0]
        count = 0
        for idx, item in enumerate(frequence):
                id_num, freq = item
                if count + freq >= fraction:
                        count = 0
                        fold_index.append(idx)
                count += freq
        
        
        K_fold_df = []
        all_groups = dataframe.groupby("ID")

        for i in range(fold_num):
                one_partition = np.array(frequence[fold_index[i]:fold_index[i+1]])
                one_partition_ids = one_partition[:, 0]
                one_partition_groups = [ all_groups.get_group(patient_ID) for patient_ID in one_partition_ids ]
                one_partition_dataset = pd.concat(one_partition_groups).reset_index(drop=True)
                K_fold_df.append(one_partition_dataset)
                
        return K_fold_df
    
    
def get_all_dataset(dataframe, augscale, fold_num):
        K_fold_df = split_K_Fold(dataframe, augscale, fold_num)
        
        
        train = ['train'] * (fold_num - 2)
        order = [ *train, 'valid', 'test']
        order = np.array(order)

        for rotate_times in range(1, fold_num+1) : 
                train_dataset, valid_dataset, test_dataset = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                train_index = np.where(order=='train')[0]
                valid_index = np.where(order=='valid')[0][0]
                test_index  = np.where(order=='test')[0][0]

                for idx in train_index:
                        train_dataset   = pd.concat( [train_dataset, K_fold_df[idx] ] ,ignore_index=False )

                valid_dataset = K_fold_df[valid_index]
                test_dataset  = K_fold_df[test_index]

                order = np.roll(order, 1)
                
                yield train_dataset, valid_dataset, test_dataset
                
                
def K_Fold_adjust_class_ratio(dataframe, argscale):
        new_dataset = pd.DataFrame()
        stage_0 = len(dataframe[dataframe["State"] == 0])
        stage_1 = len(dataframe[dataframe["State"] == 1])
        stage_2 = len(dataframe[dataframe["State"] == 2])
        stage_3 = len(dataframe[dataframe["State"] == 3])
        
        min_num = min(stage_0, stage_1, stage_2, stage_3)
        
        Class_nums = [ min_num //2 , min_num//2, min_num, min_num ]
        
        print(Class_nums)
        print(stage_0, stage_1, stage_2, stage_3)
        Stages     = [ 0, 1, 2, 3]
        
        for Stage, Class_num in zip(Stages, Class_nums):
                stage_dataset = dataframe[dataframe["State"] == Stage].reset_index(drop=True)
#                 groups = [ stage_dataset.iloc[ i:i+argscale ,:] for i in range(0, len(stage_dataset), argscale) ]
                tooth_group = stage_dataset.groupby("ID")
                groups = [ table for source, table in tooth_group ]
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
                                
                                if len(same_images) != argscale: print(same_images)
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
    
    
def K_Fold_balance_data_generator(dataframe, argscale, batch_size=32, k_fold_num=5):
        for train, valid, test in get_all_dataset(dataframe, argscale, k_fold_num):
                
                train_dataset = K_Fold_adjust_class_ratio(train, argscale=argscale)
                valid_dataset    = K_Fold_adjust_class_ratio(valid, argscale=argscale)
                test_dataset  = K_Fold_adjust_class_ratio(test , argscale=argscale)
               
                train_dataset   = shuffle(train_dataset).reset_index(drop=True)
                train_generator = make_generator(train_dataset, batch_size)

                valid_dataset   = shuffle(valid_dataset).reset_index(drop=True)
                valid_generator = make_generator(valid_dataset, batch_size)

                test_dataset    = shuffle(test_dataset).reset_index(drop=True)
                test_generator  = make_generator(test_dataset, batch_size)
                
                yield train_dataset, valid_dataset, test_dataset, train_generator, valid_generator, test_generator
                
                