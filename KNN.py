import numpy as np
import pandas as pd
import xgboost as xgb
import csv
from numpy import *
from operator import itemgetter
# from sklearn.linear_model import LogisticRegression  
# from sklearn.grid_search import GridSearchCV  
# from sklearn.preprocessing import LabelEncoder  
# from sklearn.preprocessing import StandardScaler  
# from sklearn import svm  
# from sklearn.ensemble import RandomForestClassifier  


import time

def loadTrainData():
	l = []
	r = []
	with open('train_X.csv') as data:
		lines = csv.reader(data)
		for line in lines:
			l.append(line)
		data = array(l)
	with open('train_y.csv') as lable:
		lines = csv.reader(lable)
		for line in lines:
			r.append(line)
		lable = array(r)
	# with open('train.csv') as file:
	# 	lines = csv.reader(file)
	# 	for line in lines:
	# 		l.append(line)
	# l.remove(1[0])
	# l = array(l)
	# lable = l[:,0]
	# data = l[:,1:]
	return toInt(data), toInt(lable)

def toInt(array):
	array = mat(array)
	m,n = shape(array)
	newArray = zeros((m,n))
	for i in range(m):
		for j in range(n):
			newArray[i, j] = float (array[i, j])
	return newArray

# @staticmethod
# def nomalizing(array):  
#     m,n=shape(array)
#     for i in range(m):  
#         for j in range(n):  
#             if array[i,j]== NULL:
#                 array[i,j]=0 
#     return array

def loadTestData():
    l=[]
    with open('test_final.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    # l.remove(l[0])
    data=array(l)
    return toInt(data)


# def loadTestResult():
#     l=[]
#     with open('sample_submission.csv') as file:
#          lines=csv.reader(file)
#          for line in lines:
#              l.append(line)
#     l.remove(l[0])
#     label=array(l)
#     return toInt(label[:,1])

def classify(inX, dataSet, labels, k):
    # inX=mat(inX)
    # dataSet=mat(dataSet)
    # labels=mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    # print(labels.shape)
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    print(sortedClassCount[0][0])
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','w') as myFile:
        myWriter=csv.writer(myFile)
        # myWriter.writerow('id', 'price_doc')
        for i in range(len(result)):
        	# print(i)
            myWriter.writerow([i+30474, result[i]])

def handwritingClassTest():
    trainData,trainLabel=loadTrainData()
    print(trainLabel.shape)
    print(trainData.shape)
    testData=loadTestData()
    # testLabel=loadTestResult()
    m,n=shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLabel, 5)
        resultList.append(classifierResult)
    #     print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i]))
    #     if (classifierResult != testLabel[0,i]): errorCount += 1.0
    # print ("\nthe total number of errors is: %d" % errorCount)
    # print ("\nthe total error rate is: %f" % (errorCount/float(m)))
    saveResult(resultList)

if __name__ == '__main__':
	handwritingClassTest()
	start_time = time.time()
