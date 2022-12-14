import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("../data/label/price.pickle", "rb") as f:
    label = pickle.load(f)


print(max(label))
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

plt.hist(rm_label,bins=100)
plt.show()