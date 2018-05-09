from matplotlib import pyplot as plt
import numpy as np
import os, pickle, glob
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from extractFeatures import extractFeatures

trainRatio = .8
nKNN = 10

def main():
	# fList0 = glob.glob("./t/*.mp3")
	# fList1 = glob.glob("./v/*.mp3")
	# y = np.zeros(len(fList0)+len(fList1), dtype=int)
	# y[len(fList0):] = 1
	# files = fList0 + fList1

	# # files, y = shuffle(files, y)
	# X = extractFeatures(files)

	# pickle.dump(X, open('XMFCC.pkl', 'wb'))
	# pickle.dump(y, open('yMFCC.pkl', 'wb'))
	

	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=trainRatio, shuffle=True)

	clf = neighbors.KNeighborsClassifier(nKNN, weights='uniform')
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