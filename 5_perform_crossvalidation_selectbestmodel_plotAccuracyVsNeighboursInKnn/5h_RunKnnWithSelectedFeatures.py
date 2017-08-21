#discard some features from featureMatrix in knn with n_neighbours=40





################################################################   input_output folder.... 
fs = 44100.0
M = 2048
H = 512
#time = 0                           
mfccCoeff = 'mfccCoeff'
mfccBands = 'mfccBands'
#input folders
projectfolder='/home/user/Desktop/1/'
mfccBandFolder = projectfolder	+	'4mfccextractioninpython/features/mfcc_bands/'
mfccCoffFolder = projectfolder	+	'4mfccextractioninpython/features/mfcc_coeff/'


#decide the krange
#kRange = list(range(300,320))
#wt_options=['uniform','distance']

#selected columns
selected_col=[1,2,4,6,7,9,10,11,12]  #use only these columns
folds=5
n=40


################################################################

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



from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#use only selected features
selected_f_matrix=featureMatrix[:, sel_col]


knn = KNeighborsClassifier(n_neighbors=n)
scores_knn = cross_val_score(knn, xScaled, y, cv=folds, scoring='accuracy')
print 'Mean accuracy for knn is:'
print scores_knn.mean()




