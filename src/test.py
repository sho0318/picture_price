import os
import pandas as pd
import pickle
import numpy as np

with open('../data/label/price.pickle', 'rb') as f:
    prices = pickle.load(f)

prices = np.array(prices)
mean_prices = np.mean(prices)
std_prices = max(prices) - min(prices)

rt = []
for price in prices:
    rt.append((price-mean_prices)/std_prices)

ans = {'label':rt, 'mean_std':[mean_prices, std_prices]}

with open('../data/label/price_normalize.pickle', 'wb') as f:
    pickle.dump(ans, f)
print(ans)