import mp3Util as mp3
import os
import numpy as np
import pickle

trainDir = 'trainingData/'

features = []
labels = []
for fn in os.listdir(trainDir):
	fns = fn.split('_')
	if fns[1][-1] == '7':
		continue
	if fn[0] == 't':
		labels += [1]
	elif fn[0] == 'v':
		labels += [0]
	else:
		continue
	samples = mp3.samplesFromFile(trainDir + fn)
	feature = mp3.getFeatures(samples)
	features += [feature]

feats = np.array(features)
labs = np.array(labels)
print(feats)
print(labs)
print(feats.shape, labs.shape)
pickle.dump(feats, open('featureData/X2Cs.pkl', 'wb'))
pickle.dump(labs, open('featureData/y2Cs.pkl', 'wb'))