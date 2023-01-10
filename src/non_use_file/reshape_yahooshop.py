import os
import pickle
import tempfile

import cv2
import numpy as np
import pandas as pd
import requests
from imread_from_url import imread_from_url 


def resize_fig(fig, output_path):
    resize_h = 400
    resize_w = 400

    im = cv2.resize(fig, dsize = (resize_h, resize_w))
    cv2.imwrite(output_path, im)

if __name__ == '__main__':
    with open("../data/yahoo_shop/df.pickle", 'rb') as f:
        df = pickle.load(f)

    label_list = []
    
    for i in range(len(df)):
        array = df.loc[i]
        url = array['src']
        label_list.append(array['label'])

        img = imread_from_url(url)
        print(url)

        filename = "../data/yahoo_shop/fig/{}.jpg".format(i)
        resize_fig(img, filename)
    
    with open("../data/yahoo_shop/label.pickle","wb") as f:
        pickle.dump(label_list, f) 
