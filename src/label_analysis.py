import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("../data/label/price.pickle", "rb") as f:
    label = pickle.load(f)


print(max(label))
print(np.mean(label))
print(np.std(label))

label.sort()
print(label[len(label)//2])
for i in range(30):
    label.pop(-1)

print(max(label))
print(np.mean(label))
print(np.std(label))

plt.hist(label,bins=100)
plt.show()