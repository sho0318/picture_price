import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("../data/this_is_gallery/df.pickle", "rb") as f:
    df = pickle.load(f)

print(df.head())

label = list(map(int,df['price']))
mean = np.mean(label)
std = np.std(label)

label.sort()
rm_label = []

for tmp in label:
    if tmp >= 200000:
        break
    else:
        rm_label.append(tmp)

print(max(rm_label))
print(np.mean(rm_label))
print(np.std(rm_label))
print(len(rm_label))

plt.hist(rm_label,bins=100)
plt.show()