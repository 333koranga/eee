#this compares best of knn(among n= to 50 )   and       linReg    
#find which of above two is best for us


############################################################ input and output
#input folders
projectfolder='/home/user/Desktop/1/'
mfccBandFolder = projectfolder + 'features/mfcc_bands/'
mfccCoffFolder = projectfolder  + 'features/mfcc_coeff/'


#decide the krange
n= 40  #as decided from previous code 
wt=''   #uniform or distance as deicded from previous code
folds=5




############################################################


import numpy as np
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



#s3 find accuracy for knn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=n)
scores_knn = cross_val_score(knn, xScaled, y, cv=folds, scoring='accuracy')
print 'Mean accuracy for knn is:'
print scores_knn.mean()




#s4 find accuracy for lin regression
from sklearn import linear_model
# Create linear regression object
regr = linear_model.LogisticRegression()
scores_regr = cross_val_score(regr, xScaled, y, cv=folds, scoring='accuracy')
print 'Mean accuracy for linear regression is:'
print scores_regr.mean()



print 'the better model is'
if (scores_regr  >scores_knn):
	print 'linear regression'
else:
print 'KNN'
