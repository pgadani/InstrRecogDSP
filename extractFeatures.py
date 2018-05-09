import numpy as np
import os
import librosa
from sklearn import preprocessing

def trimSound(sound, threshold = 0.001):
	maxAmp = max(sound)
	start = len(sound)
	end = 0
	for i, n in enumerate(sound):
		if n > threshold*maxAmp:
			start = i
			break
	for i, n in enumerate(sound[::-1]):
		if n > threshold*maxAmp:
			end = len(sound) - i
			break
	return start, end


def extractFeatures(fileList):
	# sr = 44100
	features = np.zeros((len(fileList), 20))
	# print(fileList)
	for i,f in enumerate(fileList):
		sound, sr = librosa.load(f)
		start, end = trimSound(sound)
		# print(f, start, end)
		tr_sound = sound[start:end]
		mfccs = librosa.feature.mfcc(y=tr_sound, sr=sr)
		if i == 0:
			print(mfccs.shape)
		avg = np.mean(mfccs, axis=1)
		features[i,:] = avg

	return features