import numpy as np
import os
import librosa
import librosa.display
from sklearn import preprocessing
import matplotlib.pyplot as plt

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


if __name__ == '__main__':

	for f in ['v/v_A3_1.mp3', 't/t_A3_1.mp3']:
		sound, sr = librosa.load(f)
		start, end = trimSound(sound)
		tr_sound = sound[start:end]
		mfccs = librosa.feature.mfcc(y=tr_sound, sr=sr)
		plt.figure()
		librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='linear')
		plt.colorbar()
		title = 'Violin' if f[0] == 'v' else 'Trumpet'
		title = title + ' A3 MFCC'
		plt.title(title)
		plt.tight_layout()
		plt.show()
		mfccs = preprocessing.scale(mfccs, axis=1)
		plt.figure()
		librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='linear')
		plt.colorbar()
		title = 'Violin' if f[0] == 'v' else 'Trumpet'
		title = title + ' A3 Scaled MFCC'
		plt.title(title)
		plt.tight_layout()
		plt.show()