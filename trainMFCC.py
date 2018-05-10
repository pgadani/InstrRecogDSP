from matplotlib import pyplot as plt
import numpy as np
import os, pickle, glob
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from extractMFCCFeatures import extractFeatures

trainRatio = .8
nKNN = 10

instrLabels = {'v': 0, 't': 1}

def main():
	files = glob.glob("./trainingData/*.mp3")
	y = np.array([f[15] for f in files])
	X = extractFeatures(files)

	pickle.dump(X, open('featureData/XMFCCt.pkl', 'wb'))
	pickle.dump(y, open('featureData/yMFCCt.pkl', 'wb'))


if __name__ == '__main__':
	main()