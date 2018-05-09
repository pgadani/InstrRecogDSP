from matplotlib import pyplot as plt
import numpy as np
import os, pickle, glob
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from newExtractFeatures import extractFeatures

trainRatio = .8
nKNN = 10

instrLabels = {'v': 0, 't': 1}

def main():
	files = glob.glob("./trainingData/*.mp3")
	y = np.zeros(len(files))

	for i, f in enumerate(files):
		y[i] = instrLabels[f[15]]


	# files, y = shuffle(files, y)
	X = extractFeatures(files)

	pickle.dump(X, open('XMFCCOurs2.pkl', 'wb'))
	pickle.dump(y, open('yMFCCOurs2.pkl', 'wb'))


if __name__ == '__main__':
	main()