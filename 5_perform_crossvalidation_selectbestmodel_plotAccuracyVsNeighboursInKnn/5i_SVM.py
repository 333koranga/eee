#find best svm model. i.e find best parameters for svm.








############################################# input folder
mfccCoeff = 'mfccCoeff'
mfccBands = 'mfccBands'
#input folders
projectfolder='/home/user/Desktop/1/'
mfccBandFolder = projectfolder	+	'4mfccextractioninpython/features/mfcc_bands/'
mfccCoffFolder = projectfolder	+	'4mfccextractioninpython/features/mfcc_coeff/'




#output folder
outputfolder=projectfolder+	'5_perform_crossvalidation_selectbestmodel_plotAccuracyVsNeighboursInKnn/svm/'


#parameters values
folds=5

kernel=['linear','rbf','poly']
C=[.1,.5,1.0,1.5,2.0]
gamma=[.001,.01,.1,1.0,10.0]
paramgrid=dict(kernel=kernel, C=C, gamma=gamma)


#####################################################################################################################################################################


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




#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
#perform cross validation on deault parameters.
from sklearn import svm
from sklearn.model_selection import cross_val_score
import time
start=time.time()

svm = svm.SVC()
#svm=svm.fit(xScaled,y)
scores_svm = cross_val_score(svm, xScaled, y, cv=folds, scoring='accuracy')
end=time.time()

print("time required in hours and in minutes")
print((end-start)/3600, (end-start)/60)


#22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
#perform cross validation on different parameters.
from sklearn import svm
from sklearn.model_selection import cross_val_score
import time
start=time.time()

meanscores=[]
allparams=[]
for ki in kernel:
	for ci in C:
		for gi in gamma:
			starti=time.time()			
			
			svm=svm.SVC(C=ci, kernel=ki, gamma=gi)
 			score=cross_val_score(svm,xScaled,y,cv=folds,scoring='accuracy')

			
			#print params
			a=[ki,ci,gi]
			b=score			
			print a
			print b
			
			#save to file
			fh=open(outputfolder + 'results' ,'a')
			fh.write(str(a)+'\n'+str(b)+'\n')
			fh.close()
	
			endi=time.time()
			print("time required in hours and in minutes for this iteration")
			print((endi-starti)/3600,(endi-starti)/60)			



end=time.time()
print("time required in hours and in minutes")
print((end-start)/3600, (end-start)/60)




#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#using grid search cv
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import time
start=time.time()

svm=svm.SVC()
grid=GridSearchCV(svm,paramgrid,cv=folds,n_jobs=-1,scoring='accuracy')
grid.fit(xScaled,y)
gridmeanscores=[result.mean_validation_score  for result in grid.grid_scores_]
print gridmeanscores

fh=open(outputfolder  +  'svmGridSearchResult'  , a)
fh.write(grid.grid.best_scores_ )
fh.close()





