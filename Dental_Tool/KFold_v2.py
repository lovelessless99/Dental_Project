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

def K_Fold_statisatic_dental_for_class(dataset):        
        dental_distribution = pd.DataFrame(index=[i for i in range(1, 33)])

        for i in range(4):
                data_frame = dataset[dataset["State"] == i]
                num = list(data_frame["tooth_num"])
                items, freqs = np.unique(num, return_counts=True)
                for item, freq in zip(items, freqs):
                        dental_distribution.loc[item,"Stage_%d"%i] = freq
        dental_distribution = dental_distribution.fillna(0)
        return dental_distribution
    
    
def K_Fold_print_class_ratio(dataframe):
        stage_0 = len(dataframe[dataframe["State"] == 0])
        stage_1 = len(dataframe[dataframe["State"] == 1])
        stage_2 = len(dataframe[dataframe["State"] == 2])
        stage_3 = len(dataframe[dataframe["State"] == 3])
        print("Class 0 : %d, Class 1 : %d, Class 2 : %d" % ( (stage_0 + stage_1), stage_2, stage_3 ))
        print("Stage 0 : %d, Stage 1 : %d, Stage 2 : %d, Stage 3 : %d" % ( stage_0, stage_1, stage_2, stage_3 ))
        
def split_K_fold(tooth_num, argscale, K_num=5):
        part = tooth_num / K_num
        ranges = [ 0, round(part), round(part*2), round(part*3), round(part*4), tooth_num]
        range_set = np.zeros((K_num, 2), int)
        
        dataset = ['train', 'train', 'train', 'valid', 'test']
        dataset_order = []
        Order_dict = collections.OrderedDict()
        
        for i in range(K_num): 
                range_set[i]  = [ranges[i], ranges[i+1]]
                dataset_order.append(dataset[-i:] + dataset[:-i])
                
        for idx, order in enumerate(dataset_order):
                Order_dict[idx] = {}
                Order_dict[idx]['train'] = [ idx for idx, x in enumerate(order) if x == 'train']
                Order_dict[idx]['valid'] = order.index('valid')
                Order_dict[idx]['test']  = order.index('test')
                
        return Order_dict, range_set * argscale
    
def K_Fold_balance_teeth_distribution(dataset, argscale, k_fold_num=5):
        for fold in range(k_fold_num):
                train_dataset, valid_dataset, testing_dataset = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
                for stage in range(4):
                        stage_dataframe = dataset[dataset["State"] == stage]
                        tooth_group = stage_dataframe.groupby("tooth_num")
                        
#                         import random
#                         groups = [df for _, df in stage_dataframe.groupby("tooth_num") ]
#                         random.shuffle(groups)
#                         stage_dataframe = pd.concat(groups).reset_index(drop=True)
#                         tooth_group = stage_dataframe.groupby("tooth_num")
                    
                        for tooth_num, tooth_table in tooth_group:
                                tooth_table = tooth_table.reset_index(drop=True)
                                
                                if len(tooth_table) % argscale != 0:
                                        print(f"========================== Warning ! not {argscale} times ==================")
                                        print(tooth_table)
                                        print(len(tooth_table))
                                        raise ValueError
                                        
                                real_tooth_amount = len(tooth_table) // argscale
                                train_end, valid_end, test_end = 0, 0, 0

                                if real_tooth_amount == 0 : continue
                                
                                if 0 < real_tooth_amount < 5:
                                        if   real_tooth_amount == 1   : train_end, valid_end, test_end = 1, 0, 0
                                        elif real_tooth_amount == 2   : train_end, valid_end, test_end = 1, 0, 1
                                        elif real_tooth_amount == 3   : train_end, valid_end, test_end = 1, 1, 1
                                        elif real_tooth_amount == 4   : train_end, valid_end, test_end = 2, 1, 1
                                        
                                        train_range = (0, int(train_end * argscale)) 
                                        valid_range = (train_range[1], int(train_range[1] + valid_end * argscale))
                                        test_range  = (valid_range[1], int(valid_range[1] + test_end  * argscale))

                                        train_dataset   = pd.concat([train_dataset  , tooth_table[train_range[0]: train_range[1] ]], ignore_index=False)
                                        valid_dataset   = pd.concat([valid_dataset  , tooth_table[valid_range[0]: valid_range[1] ]], ignore_index=False)
                                        testing_dataset = pd.concat([testing_dataset, tooth_table[test_range[0] : test_range[1]  ]], ignore_index=False)
                                
                                
                                else: 
                                        order_dict, range_set = split_K_fold(real_tooth_amount, argscale, k_fold_num)
                                        train_range = order_dict[fold]['train']
                                        valid_range = order_dict[fold]['valid']
                                        test_range  = order_dict[fold]['test']


                                        for train_range_idx in train_range:
                                                train_dataset   = pd.concat(
                                                                            [  train_dataset  , tooth_table[ range_set[train_range_idx][0]: range_set[train_range_idx][1]] 
                                                                            ], ignore_index=False
                                                                           )


                                        valid_dataset   = pd.concat([valid_dataset  , tooth_table[ range_set[valid_range][0]:  range_set[valid_range][1] ] ], ignore_index=False)
                                        testing_dataset = pd.concat([testing_dataset, tooth_table[ range_set[test_range][0] :  range_set[test_range][1]  ] ], ignore_index=False)

                yield train_dataset, valid_dataset, testing_dataset
    
    
    
def K_Fold_adjust_class_ratio(dataframe, argscale):
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
    
    
def K_Fold_balance_class_and_tooth(dataframe, argscale, k_fold_num=5):

        for train_bal_tooth, valid_bal_tooth, test_bal_tooth in K_Fold_balance_teeth_distribution(dataframe, argscale, k_fold_num):
                training_dataset = K_Fold_adjust_class_ratio(train_bal_tooth, argscale=argscale)
                valid_dataset    = K_Fold_adjust_class_ratio(valid_bal_tooth, argscale=argscale)
                testing_dataset  = K_Fold_adjust_class_ratio(test_bal_tooth , argscale=argscale)
                yield training_dataset, valid_dataset, testing_dataset
                

def K_Fold_balance_data_generator(dataframe, argscale, batch_size=32, k_fold_num=5):
    
        for train_dataset, valid_dataset, test_dataset in K_Fold_balance_class_and_tooth(dataframe, argscale, k_fold_num=k_fold_num):
                train_dataset   = shuffle(train_dataset)
                train_generator = make_generator(train_dataset, batch_size)

                valid_dataset   = shuffle(valid_dataset)
                valid_generator = make_generator(valid_dataset, batch_size)

                test_dataset    = shuffle(test_dataset)
                test_generator  = make_generator(test_dataset, batch_size)

                yield train_dataset, valid_dataset, test_dataset, train_generator, valid_generator, test_generator

