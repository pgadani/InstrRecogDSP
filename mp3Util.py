from pydub import AudioSegment
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
import math, time

N_FFT = 2**17
fs = 44100
freqGap = math.pi*fs/N_FFT

# get raw audio data as a bytestring
# check the following to make sure
# but we are assuming 2 byte samples and 1 channel
def samplesFromFile(fname):
	sound = AudioSegment.from_mp3(fname)
	rawData = sound.raw_data
	return np.fromstring(rawData, dtype=np.int16)


def plotSamples(samples, sub=None, nfft=N_FFT):
	plt.subplot(sub if sub else 121)
	plt.plot(samples)
	fftSamples = fft.rfft(samples, n=nfft)[:8192]
	plt.subplot(sub+1 if sub else 122)
	plt.plot(np.absolute(fftSamples))
	if sub is None: plt.show()


def getFeatures(samples):
	fftSamples = fft.rfft(samples, n=N_FFT)
	index = np.argmax(np.absolute(fftSamples))
	maxMag = abs(fftSamples[index])
	# print(index*freqGap, index, len(fftSamples))
	features = np.zeros(9)
	# print(len(fftSamples), index, 4*index+1)
	features[::2] = np.absolute(fftSamples)[:4*index+1:index]/maxMag
	features[1::2] = np.absolute(fftSamples)[index//2:4*index:index]/maxMag
	# print(features)
	return features


def getFeatures2(samples):
	fftSamples = fft.rfft(samples, n=N_FFT)
	index = np.argmax(np.absolute(fftSamples))
	maxMag = abs(fftSamples[index])
	maxPhase = np.angle(fftSamples[index])
	# print(index*freqGap, index, len(fftSamples))
	features = np.zeros(9)
	phases = np.zeros(9)
	# print(len(fftSamples), index, 4*index+1)
	features[::2] = np.absolute(fftSamples)[:4*index+1:index]/maxMag
	features[1::2] = np.absolute(fftSamples)[index//2:4*index:index]/maxMag
	phases[::2] = np.angle(fftSamples)[:4*index+1:index] - maxPhase
	phases[1::2] = np.angle(fftSamples)[index//2:4*index:index] - maxPhase
	features = np.concatenate((np.absolute(features), phases))
	# print(features)
	return features


if __name__ == '__main__':
	vSamples = samplesFromFile("v1.mp3")
	plotSamples(vSamples, sub=221)

	tSamples = samplesFromFile("t1.mp3")
	plotSamples(tSamples, sub=223)

	plt.show()

	vPeaks = getFeatures(vSamples)
	tPeaks = getFeatures(tSamples)

	plt.plot(vPeaks)
	plt.plot(tPeaks)
	plt.show()