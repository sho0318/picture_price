import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import re

a = "a2,bc1"
r = re.sub(r'\D', '', a)

print(r)