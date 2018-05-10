# InstrRecogDSP

A simple 2-class classifier, currently trained to distinguish audio clips of a violin note and a trumpet note. The dataset (not included here due to size) is from the London Philharmonia and can be found at <http://www.philharmonia.co.uk/explore/sound_samples/>

The simple feature set uses the magnitude (and optionally the phase) of the note at its principle frequency and 8 other half-harmonics. The data is used to train a KNN classifier and an SVM classifier.

The MFCC classifier was chosen based on "A study on feature analysis for musical instrument classification" by Jeremiah D. Deng, Christian Simmermacher, and Stephen Cranefield (referenced below). The MFCC coefficients are calculated in extractMFCCFeatures based on the tutorial from Practical Cryptography (referenced below). This was checked against the implementation in LibROSA and gave similar classification results. The visualization of MFCC is done using LibROSA. The MFCC features are also used to train a KNN classifier and an SVM classifier.

Implemented in Python 3

Dependencies:
Numpy
Scipy
Matplotlib
Sci-Kit Learn
Pydub
LibROSA (optional, for the visualization and feature extraction in visualizeFeaturesLibrosa)

References:
Dataset: http://www.philharmonia.co.uk/explore/sound_samples
Basis Research Paper: Deng, Jeremiah D., Christian Simmermacher, and Stephen Cranefield. "A study on feature analysis for musical instrument classification." IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 38.2 (2008): 429-438.
Implementation from paper: https://github.com/IvyZX/music-instrument-classifier
Tutorial on calculating MFCC: http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/