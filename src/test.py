import os
import pandas as pd
import pickle

with open('../data/label/price.pickle', 'rb') as f:
    prices = pickle.load(f)

DIR = "../data/fig"
data = []
for num in os.listdir(DIR):
    data_num = int(num.split('.')[0])
    data.append([f'{DIR}/{num}', num, prices[data_num]])

df = pd.DataFrame(data, columns=['path', 'filename', 'label'])

print(df)