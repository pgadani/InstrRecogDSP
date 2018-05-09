import mp3Util as mp3
from matplotlib import pyplot as plt
import numpy as np
import os

trainDir = 'trainingDataOld/'


def plotInstruments():
	for fn in os.listdir(trainDir):
		print(fn)
		fns = fn.split('_')
		if fns[1][-1] == '7': #ignore octaves above 6 because it would require a much larger FFT to accurately capture features
			continue
		samples = mp3.samplesFromFile(trainDir + fn)
		features = mp3.getFeatures(samples)
		plt.plot([i for i in range(len(features))], features, 'r' if fn[0]=='t' else 'b')
	plt.title('Violin vs Trumpet Features')
	plt.savefig('vtfull.png')
	plt.show()


if __name__ == '__main__':
	plotInstruments()