from Dental_Tool.Data_processing import *
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def get_group_pair_array(class_stage_dataset, fold_num):
        images_group = class_stage_dataset.groupby("source")
        group_list   = list( images_group.groups.keys() )

        sliding_window_size = len(group_list) // fold_num
        pair_array = np.zeros((fold_num, 2))

        for i in range(0, fold_num) :
                start =  sliding_window_size * i
                end   =  0

                if i == (fold_num - 1) : end = len(group_list)
                else : end =  sliding_window_size * (i + 1)

                pair = [start, end]
                pair_array[i] = pair

        return pair_array.astype(int)

def get_KFold_data(dataset_list, group_array_list, fold_num):
    
        for valid_range_idx in range(fold_num):
                fold_train_dataset, fold_valid_dataset = pd.DataFrame(), pd.DataFrame()
                
                for dataset, group_array in zip(dataset_list, group_array_list):
                        images_group = dataset.groupby("source")
                        group_list   = list( images_group.groups.keys() ) # shuffle group ? 

                        train_range_idx = [ i for i in range(fold_num) if i != valid_range_idx ]

                        train_groups = []
                        for train_idx in train_range_idx:
                                start, end = group_array[train_idx]
                                train_groups += group_list[start : end]
                        

                        valid_groups =  group_list[ group_array[valid_range_idx][0]: group_array[valid_range_idx][1] ]

                        stage_train_dataset, stage_valid_dataset = pd.DataFrame(), pd.DataFrame()

                        for idx, train_group in enumerate(train_groups): 
                                group = images_group.get_group(train_group)
                                stage_train_dataset = pd.concat([stage_train_dataset, group])

                        for idx, valid_group in enumerate(valid_groups): 
                                group = images_group.get_group(valid_group)
                                stage_valid_dataset = pd.concat([stage_valid_dataset, group])
                        
                        fold_train_dataset = pd.concat([fold_train_dataset, stage_train_dataset])
                        fold_valid_dataset = pd.concat([fold_valid_dataset, stage_valid_dataset])
                        
                print_class_ratio(fold_train_dataset)
                print_class_ratio(fold_valid_dataset)
                
                if set(fold_train_dataset.source) & set(fold_valid_dataset.source): raise ValueError
                
                fold_train_dataset = shuffle(fold_train_dataset).reset_index(drop=True)
                fold_valid_dataset = shuffle(fold_valid_dataset).reset_index(drop=True)
                
                yield fold_train_dataset, fold_valid_dataset
                
                
def K_FOLD(dataset, fold_num = 5):
        class_0_stage_0 = dataset[ (dataset["Class"] == 0) & (dataset["State"] == 0) ].reset_index(drop=True)
        class_0_stage_1 = dataset[ (dataset["Class"] == 0) & (dataset["State"] == 1) ].reset_index(drop=True)
        class_1_stage_2 = dataset[ (dataset["Class"] == 1) & (dataset["State"] == 2) ].reset_index(drop=True)
        class_2_stage_3 = dataset[ (dataset["Class"] == 2) & (dataset["State"] == 3) ].reset_index(drop=True)
        
        
        c0s0_group_array = get_group_pair_array(class_0_stage_0, fold_num)
        c0s1_group_array = get_group_pair_array(class_0_stage_1, fold_num)
        c1s2_group_array = get_group_pair_array(class_1_stage_2, fold_num)
        c2s3_group_array = get_group_pair_array(class_2_stage_3, fold_num)
        
        all_data  = [ class_0_stage_0 , class_0_stage_1 , class_1_stage_2 , class_2_stage_3    ]
        all_array = [ c0s0_group_array, c0s1_group_array, c1s2_group_array, c2s3_group_array   ]
        
        return get_KFold_data(all_data, all_array, fold_num)
    
#         class_0 = dataset[ (dataset["Class"] == 0)].reset_index(drop=True)
#         class_1 = dataset[ (dataset["Class"] == 1)].reset_index(drop=True)
#         class_2 = dataset[ (dataset["Class"] == 2)].reset_index(drop=True)
        
#         c0_group_array = get_group_pair_array(class_0, fold_num)
#         c1_group_array = get_group_pair_array(class_1, fold_num)
#         c2_group_array = get_group_pair_array(class_2, fold_num)
       
        
#         all_data  = [ class_0 , class_1 , class_2 ]
#         all_array = [ c0_group_array, c1_group_array, c2_group_array ]
#         return get_KFold_data(all_data, all_array, fold_num)