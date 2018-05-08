import mp3Util as mp3
from matplotlib import pyplot as plt
import numpy as np
import os, pickle
from sklearn import neighbors

trainRatio = .8
nKNN = 10

def main():
	X = pickle.load(open('X2ClassSimple2.pkl', 'rb'), encoding='latin1')
	y = pickle.load(open('y2ClassSimple2.pkl', 'rb'), encoding='latin1')
	print(X, y)

	class0 = X[y==0]
	class1 = X[y==1]
	
	split0 = int(len(class0)*trainRatio)
	split1 = int(len(class1)*trainRatio)

	train0 = class0[:split0]
	train1 = class1[:split1]
	test0 = class0[split0:]
	test1 = class1[split1:]

	Xtrain = np.concatenate((train0, train1))
	ytrain = np.concatenate((np.zeros(len(train0)), np.ones(len(train1))))

	Xtest = np.concatenate((test0, test1))
	ytest = np.concatenate((np.zeros(len(test0)), np.ones(len(test1))))

	clf = neighbors.KNeighborsClassifier(nKNN, weights='distance')
	clf.fit(Xtrain, ytrain)

	# correct = 0
	# # for x, y in zip(Xtest, ytest):
	# # 	print(x)
	# # 	pred = clf.predict([x])
	# # 	print(pred, y)
	# # 	if pred == y:
	# # 		correct += 1

	# print(correct, len(Xtrain))
	pred = clf.predict(Xtest)
	print(sum([1 if p == v else 0 for p, v in zip(pred, ytest)])*1.0/len(Xtest))

if __name__ == '__main__':
	main()