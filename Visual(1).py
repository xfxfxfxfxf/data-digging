
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb

import datetime
from scipy.stats import norm
import scipy as sp
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

import csv
# train = pd.read_csv('train.csv')
# filename = 'train.csv'
# with open(filename, 'r') as f:
# 	# print(f.shape)
# 	reader = csv.reader(f)
# 	header_row = next(reader)
# 	print(len(header_row))
# 	prices =[]
# 	for row in reader:
# 		price = row[291]
# 		prices.append(price)
# 	# print(prices)
# 	fig = plt.figure(dpi=128,figsize=(10,6))
# 	plt.plot(highs, c='red',linewidth=1)
lines = csv.reader(open('train.csv', 'r'))
lines = list(lines)
dataset = lines[1:len(lines)]
price_doc = []
T_T = []
for i in range(len(dataset)):
	price_doc.append(int(dataset[i][291]))
	T_T.append(dataset[i][13])
for j in range(len(T_T)):
	if T_T[j] == 'NA':
		T_T[j] = 0
# for f in range(len(T_T)):

	# if T_T[f] == 'Investment':
	# 	T_T[f] = 1
	# if T_T[f] == 'OwnerOccupier':
	# 	T_T[f] = 2
	# if int(T_T[f]) < 20:
		# T_T[f] = 20
f1 = plt.figure(1)
plt.scatter(T_T, price_doc)
plt.xlabel('area_m')
plt.ylabel('price_doc')
plt.show()

