#objective1 : Plot different features and analyze for feature selection





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


#output folder
outputfolder= projectfolder +   '5_perform_crossvalidation_selectbestmodel_plotAccuracyVsNeighboursInKnn/FeatureSelectionPlots/'

###############################################################################################






import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd





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




#in dataframe each row and each column has a name. so give name to each row and each column
row=[]
for i in range(featureMatrix.shape[0]):
	row.append("Row "+str(i))


col=[]
for i in range(featureMatrix.shape[1]-1):
	col.append("Coeffcient "+str(i))


col.append('Label')   #name all features as coeff 0, coeff 2, coeff 3... coeff 12 and last column as label

#convert feature matrix to dataframe: data is the real data,  index is the name of rows,   column is the name of columns
f=pd.DataFrame(data=featureMatrix[:,:],index=row,columns=col)

#analyze by plotting. x_vars contain the name of columns to plot,  y_vars contain the name of col to compare to (it is usually the output column).
#x_vars and y_vars should be in the f data frame.
for i in range(featureMatrix.shape[1]-1):
	sns.pairplot(f,x_vars=col[i],y_vars=col[-1],size=7,aspect=.7,kind='reg')
	plt.savefig(outputfolder+'feature'+str(i))
	#plt.show()












