import numpy as np
import pickle

with open('../data/meta_log.pkl', 'rb') as f:
    d = pickle.load(f)

print(d)
