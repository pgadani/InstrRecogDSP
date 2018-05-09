from matplotlib import pyplot as plt
import numpy as np
import os, pickle
from sklearn import neighbors
from sklearn.model_selection import train_test_split

trainRatio = .8
nKNN = 1
featType = 'MFCC' # 'Simple9', 'Simple18'

def main():
	if featType = 'Simple9':
		fileSet = '2Cs'
	elif featType = 'Simple18':
		fileSet = '2Cs2'
	else:
		fileSet = 'MFCC'

	X = pickle.load(open('featureData/X' + fileSet + '.pkl', 'rb'), encoding='latin1')
	y = pickle.load(open('featureData/y' + fileSet + '.pkl', 'rb'), encoding='latin1')

	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=trainRatio, shuffle=True)

	clf = neighbors.KNeighborsClassifier(nKNN, weights='distance')
	clf.fit(Xtrain, ytrain)

	pred = clf.predict(Xtest)
	print(sum([1 if p == v else 0 for p, v in zip(pred, ytest)])*1.0/len(Xtest))

if __name__ == '__main__':
	main()