from Dental_Tool.Data_processing import make_generator
from sklearn.utils import shuffle
import collections
import pandas as pd
import numpy as np
import json
import os 

def load_json(data_list, interdental=False):
            filter_fun = lambda x : { path: max(list(map(int, state))) for path, state in x.items() if max(list(map(int, state))) >= 0 }
               
            interdental_fun = lambda x : { path: state for path, state in x.items() if state >= 0 }
            
            results = collections.OrderedDict()
            all_filtering_data, all_keys = [], []
            
            for dataset_path in data_list:
                        mapping_data = json.load(open(dataset_path , "r"))
                        filter_data  = filter_fun(mapping_data) if not interdental else interdental_fun(mapping_data)
                        all_filtering_data.append(filter_data)
                        all_keys.append( list(filter_data.keys()) )
            
            for keys in zip(*all_keys):
                    for key,  data in zip(keys, all_filtering_data):
                            results[key] = data[key]        
            return results
                        
def json_2_dataframe_PBL(data, mode=None):
        PBL_Columns = ["Path", "State", "Class"]
        dataframe = pd.DataFrame(columns=PBL_Columns)
        data_dict, counter = collections.OrderedDict(), 0
        
        molar    = [ 1, 2 , 3 , 14, 15, 16, 17, 18, 19, 30, 31, 32]
        premolar = [ 4, 5 , 12, 13, 20, 21, 28, 29 ]
        canine   = [ 6, 11, 22, 27                 ]
        incisor  = [ 7, 8 , 9 , 10, 23, 24, 25, 26 ]
        
        all_molar = molar + premolar
        
        less_data  = [1, 16, 17, 32]
        
        for path, state in data.items():
                item = { "Path": path, "State": state, "Class": state-1 if state >= 1 else 0  }
                path_split = path.split("_")
                
#                 in_dir = path.split("/")[2]

                NN_IDX = 0
                for idx, i in enumerate(path_split):
                        if i == "NN":
                            NN_IDX = idx
                            break

                original, source = "", ""
                if NN_IDX == 0: 
                        source = "_".join(path_split[-6:-1]) 
                        original = "_".join(path_split[-6:-2])
                        
                else: 
                        source = "_".join(path_split[NN_IDX:-1])
                        original = "_".join(path_split[NN_IDX:-2])
                 
                item["ori_src"] = original
                item["source"] = source
                item["tooth_num"] = int(path_split[-2])
                
                if item["tooth_num"] in molar     : item["tooth_type"] = 0
                elif item["tooth_num"] in premolar: item["tooth_type"] = 1
                elif item["tooth_num"] in canine  : item["tooth_type"] = 2
                elif item["tooth_num"] in incisor : item["tooth_type"] = 3
                else : item["tooth_type"] = -99
                    
                cond_1 = (mode == "molar"    ) and (item["tooth_num"] not in molar    )
                cond_2 = (mode == "premolar" ) and (item["tooth_num"] not in premolar )
                cond_3 = (mode == "canine"   ) and (item["tooth_num"] not in canine   )
                cond_4 = (mode == "incisor"  ) and (item["tooth_num"] not in incisor  )
                cond_5 = (mode == "all_molar") and (item["tooth_num"] not in all_molar)
                if cond_1 or cond_2 or cond_3 or cond_4 or cond_5: continue
                    
                item["angle"] = int(path_split[-1].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe

def json_2_dataframe_PBL_inderdental(data, mode=None):
        PBL_Columns = ["Path", "State", "Class"]
        dataframe = pd.DataFrame(columns=PBL_Columns)
        data_dict, counter = collections.OrderedDict(), 0
        
        molar    = [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
        premolar = [ 4, 5, 12, 13, 20, 21, 28, 29 ]
        canine   = [ 6, 11, 22, 27                 ]
        incisor  = [ 7, 8 , 9 , 10, 23, 24, 25, 26 ]
        all_molar = molar + premolar
        
        less_data  = [1, 16, 17, 32]
        
        for path, state in data.items():
                item = { "Path": path, "State": state, "Class": state-1 if state >= 1 else 0  }
                path_split = path.split("_")
                
#                 in_dir = path.split("/")[2]

                NN_IDX = 0
                for idx, i in enumerate(path_split):
                        if i == "NN":
                            NN_IDX = idx
                            break
                
                Patrica_IDX = 0
                if NN_IDX == 0:
                        for idx, i in enumerate(path_split):
                                if "Patrica" in i:
                                    Patrica_IDX = idx
                                    break
                
                
                original, source = "", ""
                if NN_IDX == 0: 
                        source = "_".join(path_split[-7:-2]) 
                        original = "_".join(path_split[-7:-3])
                else: 
                        source = "_".join(path_split[NN_IDX:-2])
                        original = "_".join(path_split[NN_IDX:-3])
                
#                 if ' ' == path_split[-4][1]:  ID = "_".join(path_split[-5:-3])
#                 else  : ID = path_split[-4]

                if NN_IDX != 0:
                        if ' ' == path_split[NN_IDX-1][1]:  ID = "_".join(path_split[NN_IDX-2:NN_IDX])
                        else  : ID = path_split[NN_IDX-1]
                
                else: 
                        if ' ' == path_split[Patrica_IDX-1][1]:  ID = "_".join(path_split[Patrica_IDX-2:Patrica_IDX])
                        else  : ID = path_split[Patrica_IDX-1]
                
                item["ID"] = ID
                
                item["ori_src"] = original
                item["source"] = source
                item["tooth_num"] = int(path_split[-3])
                
                
                if item["tooth_num"] in molar     : item["tooth_type"] = 0
                elif item["tooth_num"] in premolar: item["tooth_type"] = 1
                elif item["tooth_num"] in canine  : item["tooth_type"] = 2
                elif item["tooth_num"] in incisor : item["tooth_type"] = 3
                else : item["tooth_type"] = -99
                    
                item["side"] = source + "_" + path[-5]
                
                
                cond_1 = (mode == "molar"    ) and (item["tooth_num"] not in molar    )
                cond_2 = (mode == "premolar" ) and (item["tooth_num"] not in premolar )
                cond_3 = (mode == "canine"   ) and (item["tooth_num"] not in canine   )
                cond_4 = (mode == "incisor"  ) and (item["tooth_num"] not in incisor  )
                cond_5 = (mode == "all_molar") and (item["tooth_num"] not in all_molar)
                if cond_1 or cond_2 or cond_3 or cond_4 or cond_5: continue
                    
                item["angle"] = int(path_split[-2].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe
    
def init_directory(dir_name):
        if not os.path.isdir(dir_name): os.makedirs(dir_name)
            
            
def prepared_data(path, class_num, fold_num, batch_size=64):
    
         for fold in range(1, fold_num+1):
                   
                train_dataset = pd.read_csv(f"{path}/Fold_{fold}/train_dataset.csv")
                valid_dataset = pd.read_csv(f"{path}/Fold_{fold}/valid_dataset.csv")
                test_dataset  = pd.read_csv(f"{path}/Fold_{fold}/test_dataset.csv")
               
                train_dataset   = shuffle(train_dataset).reset_index(drop=True)
                train_generator = make_generator(train_dataset, batch_size)

                valid_dataset   = shuffle(valid_dataset).reset_index(drop=True)
                valid_generator = make_generator(valid_dataset, batch_size)

                test_dataset    = shuffle(test_dataset).reset_index(drop=True)
                test_generator  = make_generator(test_dataset, batch_size)
                
                yield train_dataset, valid_dataset, test_dataset, train_generator, valid_generator, test_generator