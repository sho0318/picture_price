import numpy as np
import pickle

with open('../data/yahoo_shop/label.pickle', 'rb') as f:
    label = pickle.load(f)

label = np.array(list(map(float,(label))))
mean = label.mean()
std = label.std()

print(mean)
print(std)

dist = {'label': (label-mean)/std, 'mean_std': [mean, std]}

with open('../data/yahoo_shop/normalize_label.pickle', 'wb') as f:
    pickle.dump(dist, f)