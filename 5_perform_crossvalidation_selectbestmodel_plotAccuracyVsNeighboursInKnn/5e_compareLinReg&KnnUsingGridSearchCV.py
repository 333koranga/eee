#this compares best of knn(among n= to 50 )   and       linReg    
#find which of above two is best for us


############################################################ input and output
#input folders
mfccBandFolder = '/home/user/Desktop/1/features/mfcc_bands/'
mfccCoffFolder = '/home/user/Desktop/1/features/mfcc_coeff/'


#decide the krange
n=   #as decided from previous code 
wt=''   #uniform or distance as deicded from previous code





############################################################



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
knn = KNeighborsClassifier(n_neighbors=n,weights=wt)
scores_knn = cross_val_score(knn, xScaled, y, cv=10, scoring='accuracy')
print 'Mean accuracy for knn is:'
print scores.mean()




#s4 find accuracy for lin regression
from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
scores_regr = cross_val_score(regr, xScaled, y, cv=10, scoring='accuracy')
print 'Mean accuracy for linear regression is:'
print scores.mean()



print 'the better model is'
if (scores_regr  >scores_knn):
	print 'linear regression'
else:
	print 'KNN'

