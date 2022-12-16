import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("../data/yahoo_shop/df.pickle", "rb") as f:
    df = pickle.load(f)

print(df.head())
label = list(map(int,df['label']))
mean = np.mean(label)
std = np.std(label)

label.sort()
rm_label = []

for tmp in label:
    if tmp >= mean + std*3:
        break
    else:
        rm_label.append(tmp)

print(max(rm_label))
print(np.mean(rm_label))
print(np.std(rm_label))

plt.hist(label,bins=100)
plt.show()