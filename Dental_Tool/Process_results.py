import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import heapq
import glob
import os

def plot_result(history, output_dir):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(16,5))
        
        plt.subplot(121)
        plt.title("Loss")
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label = 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        
        plt.subplot(122)
        plt.title("Accuracy")
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1.5])
        plt.legend(loc='lower right')
        
#         target_dir = "Results/%s" % (output_dir)
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        plt.savefig("%s/history.png" % (output_dir))
        plt.show()
        
        
        
def plot_confusion_matrix(confusion_matrix, classes, acc, output_dir, idx):
        df_cm = pd.DataFrame(confusion_matrix, range(classes), range(classes))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='YlGnBu', fmt='g') # font size
        plt.title("Confusion Matrix (Acc = {:5.2f}%)".format(acc))
        plt.xlabel('prediction' , fontsize=18)
        plt.ylabel('ground truth', fontsize=18)
        
        target_dir = "%s/cmxs" % (output_dir)
        if not os.path.isdir(target_dir): os.makedirs(target_dir)
        plt.savefig("{}/cmx_acc_{:5.2f}.png".format(target_dir, acc), transparent=True, bbox_inches='tight', pad_inches=0 )
        
        plt.show()
        
def get_k_top_value(param_path, k_th=3):
        names, values = [], []
        for param in glob.iglob( "%s/*" % param_path  ):
                names.append(param)
                values.append(float(param.split(".")[1]))
        top_n_pair =  heapq.nlargest(k_th, zip(values, names))
        ans = [ pair[1] for pair in top_n_pair ]
        return ans         
        
        
def vote(dataframe, results, model_name_list):
        for idx, name in enumerate(model_name_list):
                dataframe[name] = results[idx]
        
        vote_results = [0] * len(dataframe)
        results = np.array(results).T
        no_majority_num = 0
        for idx, item in enumerate(results):
                item = list(item)
                ans = -99
                if item[0] == item[1] == item[2]: ans = item[0]
                elif item[0] == item[1] or item[1] == item[2] or item[0] == item[2]: ans = max(set(item), key = item.count)
                else: 
#                         print("No Majority!")
                        no_majority_num += 1
#                         ans = np.random.randint(3, size=1)[0]
                        ans = 1
                vote_results[idx] = ans
        print("# of no majority: %d" % no_majority_num )
        dataframe["Predict"] = vote_results
        return dataframe
    
def statistic(dataframe):
        results = pd.DataFrame(columns=["Class 0", "Class 1", "Class 2"])
        for i in range(4):
                data = {
                    "Class 0": len(dataframe[(dataframe["State"]==i) & (dataframe["Predict"]==0)]),
                    "Class 1": len(dataframe[(dataframe["State"]==i) & (dataframe["Predict"]==1)]),
                    "Class 2": len(dataframe[(dataframe["State"]==i) & (dataframe["Predict"]==2)]),
                }
                
                total = data["Class 0"] + data["Class 1"] + data["Class 2"]
                if i <= 1: Acc = data["Class 0"] / total * 100
                if i == 2: Acc = data["Class 1"] / total * 100
                if i == 3: Acc = data["Class 2"] / total * 100

                data["Acc"] = "{}%".format(int(round(Acc,0)))
                results = results.append(data, ignore_index=True)
        return results