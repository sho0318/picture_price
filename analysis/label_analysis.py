import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

with open("../data/this_is_gallery/df.pickle", "rb") as f:
    df = pickle.load(f)

print(df)


label = list(map(float,df['price']))
mean = np.mean(label)
std = np.std(label)

label.sort()
rm_label = []

for tmp in label:
    if tmp >= 100000:
        break
    else:
        rm_label.append(tmp)

df_rm = pd.DataFrame(rm_label)

print('--------------')
print(max(rm_label))
print(min(rm_label))
print(np.mean(rm_label))
print(np.std(rm_label))
print(len(rm_label))
print("歪度:", df_rm.skew())
print("尖度:", df_rm.kurt())

plt.hist(rm_label)
plt.show()

