## for  SVM

import numpy as np
x=np.array(([1],[1],[1],[1],[1],[1],[1],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]))
#x.reshape(-1,1)
y=np.array( [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1] )






kernel=['linear','rbf','poly']
C=[.1,.5,1,1.5]
gamma=[.001,.01,.1,1.0,10.0]
paramgrid=dict(kernel=kernel, C=C, gamma=gamma)
folds=5



from sklearn import svm
from sklearn.grid_search import GridSearchCV
import time
s=time.time()


svm=svm.SVC()
grid = GridSearchCV(svm, paramgrid, cv=folds, n_jobs=-1, scoring='accuracy')#-1 means number of cpu cores
grid.fit(x,y)
gridmeanscores=[result.mean_validation_score  for result in grid.grid_scores_]
print gridmeanscores


