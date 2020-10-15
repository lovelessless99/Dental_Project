import collections
import pandas as pd
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
        
        molar = [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
        premolar = [ 4, 5, 12, 13, 20, 21, 29, 30 ]
        
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
                
                if (mode == "molar" and (item["tooth_num"] not in molar)   ) or (mode == "premolar" and (item["tooth_num"] not in premolar) ): continue 
                    
                item["angle"] = int(path_split[-1].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe

def json_2_dataframe_PBL_inderdental(data, mode=None):
        PBL_Columns = ["Path", "State", "Class"]
        dataframe = pd.DataFrame(columns=PBL_Columns)
        data_dict, counter = collections.OrderedDict(), 0
        
        molar = [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
        premolar = [ 4, 5, 12, 13, 20, 21, 29, 30 ]
        
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
                
                if (mode == "molar"    and item["tooth_num"] not in molar   ) or (mode == "premolar" and item["tooth_num"] not in premolar): continue 
                
                item["angle"] = int(path_split[-2].split(".")[0])
                
                data_dict[counter] = item
                counter += 1        
        dataframe = dataframe.from_dict(data_dict, "index")
        return dataframe
    
def init_directory(dir_name):
        if not os.path.isdir(dir_name): os.makedirs(dir_name)