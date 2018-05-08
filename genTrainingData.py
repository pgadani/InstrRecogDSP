import os, shutil

srcDir = 'trumpet/'
dstDir = 'trainingData/'

# arco-normal for violin, normal for trumpet
identifier = 'arco-normal' if srcDir == 'violin/' else 'normal'

volumeMap = {'forte': 'f', 'pianissimo': 'p', 'fortissimo': 's'}

# The training data is split as 1_2_3_4_5.mp3, where:
#	1 - instrument
#	2 - note
#	3 - number
#	4 - volume
#	5 - style
def genTrainingData(directory):
	count = 0
	maxCount = 10000
	for fn in os.listdir(directory):
		# print(fn)
		if fn.endswith('.mp3'):
			if count >= maxCount:
				break
			fns = fn[:-4].split('_')
			if fns[3] not in volumeMap:
				continue
			if fns[4] != identifier:
				continue
			if not fns[2].isdigit():
				continue
			newFn = fns[0][0] + '_' + fns[1] + '_' + fns[2] + '_' + volumeMap[fns[3]] + '.mp3'
			print(newFn)
			shutil.copy2(directory+fn, dstDir+newFn)
			count+=1



if __name__ == '__main__':
	genTrainingData(srcDir)