# show_pkl.py

import pickle
import pprint

path = '../log/exp/exp_behavior_Generate_Traffic_seed_0/eval_results/records_ttc.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
print('data有', len(data), '条TTC数据')
print(type(data))
for i in range(len(data)):
    print(data[i])
