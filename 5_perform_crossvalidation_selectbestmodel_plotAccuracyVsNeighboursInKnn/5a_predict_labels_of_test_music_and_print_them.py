#before this code calculate mfccs for test files. store to some folder which will be input folder
#this code takes mfccs(of test music files) from some input folder and predicts labels (vocal/nonvocal) for each frame in a mfcc file.
#it then outputs the labels without saving them




import os
from essentia import *
from essentia.standard import *
import essentiaSpecGram as essSpec
import matplotlib.pyplot as plt
import numpy as np
from pylab import plot, show, figure
plt.rcParams['agg.path.chunksize'] = 10000

###############################################  input and output folders
#input_folder for bollywoodmusci.npy

print("1")
mfccCoeff = 'mfccCoeff'
mfccBands = 'mfccBands'
#folders to store mfccBand and mfccCoff
mfccBandFolder = '/home/user/Desktop/1/features/mfcc_bands/'
mfccCoffFolder = '/home/user/Desktop/1/features/mfcc_coeff/'

#input_folder for test_file_mfccs (mfccs of test files )
test_mfccfile_path='/home/user/Desktop/1/test/features/mfcc_coeff/'







#----1 import scikit-learn package
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics




#-----2 provide filepath for bollywood music and load the matrix
featureMatrix = np.load(mfccCoffFolder + "bollywoodmusic" + ".npy")




#---------------   3 normalize the bollywood music
#X will have mfccmatrix
X = featureMatrix[:, :-1] 
#y will have labels
y = featureMatrix[:, -1]    
print("2")
# Preprocessing the feature vectors
xScaled = np.copy(X)
for i in range(X.shape[1]):
    xScaled[:, i] = (X[:, i] - np.mean(X[:, i])) / np.abs(np.max(X[:, i]) - np.min(X[:, i]))



#-------4 initialize knn and knn.fit
knn = KNeighborsClassifier(n_neighbors=19) 
'''  what to put the values of n_neighbours'''
knn.fit(xScaled,y)

print("3")


#-------provide filepath for testfiles mfccs  and load the filenames
test_mfccfiles_list= [files_i 
            for files_i in os.listdir(test_mfccfile_path)
            if files_i.endswith('.npy')]
print("4")




#----------predict all test mfccs
for mfccfilename in test_mfccfiles_list:
	#load the mfcc
	X=np.load(test_mfccfile_path+mfccfilename)
	xScaled = np.copy(X)
	for i in range(X.shape[1]):
	    xScaled[:, i] = (X[:, i] - np.mean(X[:, i])) / np.abs(np.max(X[:, i]) - np.min(X[:, i]))
	print("5")
	#------predict the mfcc
	result=knn.predict(xScaled)
	print("print results for mfcc file: "+mfccfilename)
	print(result)











