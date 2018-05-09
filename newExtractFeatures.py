import numpy as np
from numpy.fft import rfft
from scipy.fftpack import dct
from sklearn import preprocessing

import mp3Util as mp3

from speechRecog import mfcc

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


def mfccOurs(sound, sr, frameSize=0.023, frameStride=0.01, nfft=512, nfilt=26, nCoeff=12):
	sigLen = sound.shape[0]
	frameLen = int(frameSize*sr)
	frameStep = int(frameStride*sr)
	frameCount = int(np.ceil(abs(sigLen - frameLen)/frameStep))
	paddedSig = np.zeros(frameCount*frameStep + frameLen, dtype=np.int16)
	paddedSig[:sigLen] = sound
	frames = np.zeros((frameCount, frameLen))
	for i in range(frameCount):
		frames[i,:] = np.hamming(frameLen) * paddedSig[i*frameStep:i*frameStep+frameLen]

	# Do FFT to get power spectrum
	mag = np.absolute(rfft(frames, nfft))
	powerSpec = np.square(mag) *1.0/nfft

 	# Bounds of frequencies in Mel scale
	lowMel = 0
	highMel = (2595 * np.log10(1 + (sr / 2) / 700))
	melPts = np.linspace(lowMel, highMel, nfilt + 2) # Equally spaced points Mel scale
	hzPts = (700 * (10**(melPts / 2595) - 1))  # Same points in hz
	fftBins = np.floor((nfft + 1) * hzPts / sr)

	fbank = np.zeros((nfilt, nfft//2 + 1))
	for m in range(1, nfilt + 1):
		left = int(fftBins[m - 1])
		center = int(fftBins[m])
		right = int(fftBins[m + 1]) 

		for i in range(left, center):
			fbank[m - 1, i] = (i - fftBins[m - 1]) / (fftBins[m] - fftBins[m - 1])
		for i in range(center, right):
			fbank[m - 1, i] = (fftBins[m + 1] - i) / (fftBins[m + 1] - fftBins[m])
	filtSig = np.dot(powerSpec, fbank.T)
	filtSig = np.where(filtSig == 0, np.finfo(float).eps, filtSig)  # Because log of 0 is bad
	filtSig = 20 * np.log10(filtSig)  # dB
	mfcc = dct(filtSig, type=2, axis=1, norm='ortho')[:, 1 : (nCoeff + 1)] # Keep 2-13
	return mfcc



def extractFeatures(fileList, nfeat=20):
	sr = 22050
	features = np.zeros((len(fileList), nfeat))
	# print(fileList)
	for i,f in enumerate(fileList):
		sound = mp3.samplesFromFile(f)
		start, end = trimSound(sound)
		tr_sound = sound[start:end]
		# mfccs = mfcc(tr_sound, samplerate=sr, winlen=0.023)
		mfccs = mfccOurs(tr_sound, sr, nCoeff=nfeat)
		# print(mfccs.shape)
		avg = np.mean(mfccs, axis=0)
		features[i,:] = avg

	return features