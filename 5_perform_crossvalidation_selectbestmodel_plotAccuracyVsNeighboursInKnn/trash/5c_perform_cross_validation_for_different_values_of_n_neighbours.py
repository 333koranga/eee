#this aims to find best knn model (among n=5 to n=50) using Grid Search CV.



#this contains the last portion of mfccfeatureextraction_v2
#it starts with machine learning part

#this code performs cross-validation and cal the mean accuracy scores for a range of n_neighbours

#step1 load bollywood mysic
#2 normalize
#3 specify krange range
#4 perform cross validation
#5 results are output to screen , not stored





import os
from essentia import *
from essentia.standard import *
#import essentiaSpecGram as essSpec
import matplotlib.pyplot as plt
import numpy as np



################################################################   input_output folder.... 
fs = 44100.0
M = 2048
H = 512
#time = 0                           
mfccCoeff = 'mfccCoeff'
mfccBands = 'mfccBands'
#input folders
mfccBandFolder = '/home/user/Desktop/1/features/mfcc_bands/'
mfccCoffFolder = '/home/user/Desktop/1/features/mfcc_coeff/'


#decide the krange
kRange = list(range(5, 7))
wt_options=['uniform','distance']


################################################################
#s0 this is to print execution time of this code
import time
start_time = time.time()


#s1 load bollywoodMusic.npy
print("1")
featureMatrix = np.load(mfccCoffFolder + "bollywoodmusic" + ".npy")



#s2 normalize
#X will have mfccmatrix
X = featureMatrix[:, :-1] 
#y will have labels
y = featureMatrix[:, -1]    
print("2")
# Preprocessing the feature vectors
xScaled = np.copy(X)
for i in range(X.shape[1]):
    xScaled[:, i] = (X[:, i] - np.mean(X[:, i])) / np.abs(np.max(X[:, i]) - np.min(X[:, i]))




'''
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#split for cross validation
Xtrain, Xtest, yTrain, yTest = train_test_split(xScaled, y, random_state=4)
print("2a")
from sklearn.cross_validation import cross_val_score
print("3")

kScore = []
for k in kRange:
	print("4")   
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, xScaled, y, cv=5, scoring='accuracy')
	kScore.append(scores.mean())
print("6")
print(kScore)
'''

# Fast implementation of the algorithms    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
parmGrid = dict(n_neighbors=kRange,weights=wt_options)#in vid 8
grid = GridSearchCV(knn, parmGrid, cv=10, n_jobs=-1, scoring='accuracy')#-1 means number of cpu cores
grid.fit(xScaled, y)



#print the execution time of this code
print("--- %s seconds required for n=5 to n=6---" % (time.time() - start_time))


#grid.grid_scores_  for full output
grid.best_score_
grid.best_params_
gridmeanscores=[result.mean_validation_score  for result in grid.grid_scores_]#in vid 8
print gridmeanscores


plt.plot(kRange,gridmeanscores)
plt.xlabel('values of k for KNN')
plt.ylabel('Cross-validation accuracy')
show()

#examine the best model
print grid.best_score_
print grid.best_params_
print grid.best_estimator_


