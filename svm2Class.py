from matplotlib import pyplot as plt
import numpy as np
import os, pickle
from sklearn import svm
from sklearn.model_selection import train_test_split

trainRatio = .8
dataDir = '2Cs2.pkl'

def main():
	X = pickle.load(open('X' + dataDir, 'rb'), encoding='latin1')
	y = pickle.load(open('y' + dataDir, 'rb'), encoding='latin1')

	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=trainRatio, shuffle=True)

	clf = svm.SVC(kernel='poly')
	clf.fit(Xtrain, ytrain)

	pred = clf.predict(Xtest)
	print(sum([1 if p == v else 0 for p, v in zip(pred, ytest)])*1.0/len(Xtest))

if __name__ == '__main__':
	main()