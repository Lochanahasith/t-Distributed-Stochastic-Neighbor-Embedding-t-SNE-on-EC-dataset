import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn import preprocessing
import matplotlib.pyplot as plt


datawithnan= pd.read_csv('endometrial.csv')


data1 = datawithnan.dropna() # remove missing values


