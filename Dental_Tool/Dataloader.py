import collections
import pandas as pd
import json
import os 

def load_json(data_li, interdental=False):
            normal_path, normal_flip_path, clahe_path, clahe_flip_path = data_li
            
            mapping_clahe_data = json.load(open(clahe_path , "r"))
            mapping_data = json.load(open(normal_path, "r"))
            mapping_clahe_data_Flip = json.load(open(clahe_flip_path, "r"))
            mapping_data_Flip = json.load(open(normal_flip_path, "r"))

            filter_fun = lambda x : { path: max(list(map(int, state))) for path, state in x.items() if max(list(map(int, state))) >= 0 }
               
            interdental_fun = lambda x : { path: state for path, state in x.items() if state >= 0 }
            
            filter_normal_data       = filter_fun(mapping_data) if not interdental else interdental_fun(mapping_data)
            
            filter_normal_data_Flip  =  filter_fun(mapping_data_Flip) if not interdental else interdental_fun(mapping_data_Flip)

            filter_clahe_data        = filter_fun(mapping_clahe_data) if not interdental else interdental_fun(mapping_clahe_data)
            
            filter_clahe_data_Flip   = filter_fun(mapping_clahe_data_Flip) if not interdental else interdental_fun(mapping_clahe_data_Flip)


            filter_data = collections.OrderedDict()

            key_pair_gen = zip(filter_normal_data.keys(), 
                               filter_normal_data_Flip.keys(), 
                               filter_clahe_data.keys(), 
                               filter_clahe_data_Flip.keys())

            for normal, normal_flip, clahe, clahe_flip in key_pair_gen:
                    filter_data[normal]      = filter_normal_data[normal]
                    filter_data[normal_flip] = filter_normal_data_Flip[normal_flip]
                    filter_data[clahe]       = filter_clahe_data[clahe]
                    filter_data[clahe_flip]  = filter_clahe_data_Flip[clahe_flip]
                    
            return filter_data
                        
def json_2_dataframe_PBL(data):
        PBL_Columns = ["Path", "State", "Class"]
        dataframe = pd.DataFrame(columns=PBL_Columns)
        data_dict, counter = collections.OrderedDict(), 0
        
        multi_root = [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
        less_data  = [1, 16, 17, 32]
        
        for path, state in data.items():
                item = { "Path": path, "State": state, "Class": state-1 if state > 1 else 0  }
                path_split = path.split("_")
                
#                 in_dir = path.split("/")[2]

                NN_IDX = 0
                for idx, i in enumerate(path_split):
                        if i == "NN":
                            NN_IDX = idx
                            break
                source = ""
                if NN_IDX == 0: source = "_".join(path_split[-6:-1]) 
                else: source = "_".join(path_split[NN_IDX:-1])
                
                item["source"] = source
                item["tooth_num"] = int(path_split[-2])
                item["angle"] = int(path_split[-1].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe

def json_2_dataframe_PBL_inderdental(data):
        PBL_Columns = ["Path", "State", "Class"]
        dataframe = pd.DataFrame(columns=PBL_Columns)
        data_dict, counter = collections.OrderedDict(), 0
        
        multi_root = [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
        less_data  = [1, 16, 17, 32]
        
        for path, state in data.items():
                item = { "Path": path, "State": state, "Class": state-1 if state > 1 else 0  }
                path_split = path.split("_")
                
#                 in_dir = path.split("/")[2]

                NN_IDX = 0
                for idx, i in enumerate(path_split):
                        if i == "NN":
                            NN_IDX = idx
                            break
                source = ""
                if NN_IDX == 0: source = "_".join(path_split[-7:-2]) 
                else: source = "_".join(path_split[NN_IDX:-2])
                
                item["source"] = source
                item["tooth_num"] = int(path_split[-3])
                item["angle"] = int(path_split[-2].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe
    
def init_directory(dir_name):
        if not os.path.isdir(dir_name): os.makedirs(dir_name)