import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
from collections import Counter
import datetime
from scipy.stats import norm
import scipy as sp
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

import csv
train = pd.read_csv('train.csv')
filename = 'train.csv'
with open(filename, 'r') as f:
	# print(f.shape)
	reader = csv.reader(f)
	# header_row = next(reader)
	# print(len(header_row))
	# prices =[]
	# for row in reader:
	# 	price = row[291]
	# 	prices.append(price)
	column = [row[13] for row in reader]
	print(column)
	
	# count = Counter(column)
	# print(count)
	# print(prices)
# 	fig = plt.figure(dpi=128,figsize=(10,6))
# 	plt.plot(highs, c='red',linewidth=1)