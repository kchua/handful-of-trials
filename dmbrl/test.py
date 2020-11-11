import numpy as np
from matplotlib import pyplot as plt

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

a = np.random.uniform(size=(3,3,3))
print(a)
a = shuffle_rows(a)
print(a)