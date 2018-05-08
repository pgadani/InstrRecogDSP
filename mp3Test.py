from pydub import AudioSegment
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

N_FFT = 2**17
fs = 44100

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

vSamples = samplesFromFile("v1.mp3")
plotSamples(vSamples, sub=321)

fSamples = samplesFromFile("f1.mp3")
plotSamples(fSamples, sub=323)

tSamples = samplesFromFile("t1.mp3")
plotSamples(tSamples, sub=325)

plt.show()