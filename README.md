[![Platform](https://img.shields.io/badge/Platform-Tensorflow-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.5-green.svg)]()

# Arrthymias Detection Based on Electrocardiogram Using Deep Autoencoder

AD (Arrhythmias Detection) is a repo that investigates variant autoencoder(AE) in feature extraction, and applies them to classify from Electrocardiogram(ECG) to arrhythmias.

Now AD includes 4 AE as feature extractor, and classify by using Random Forest. 

# Arrhythmias Data

Followed by the MIT-BIH Arrhythmias Database

- [MITDB](https://physionet.org/physiobank/database/mitdb/)   
This database is described in
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
Also, more functions are updating.

Citation:
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).

# Requirements

Before you start testing, following requirements are needed.

- Python3.6.5
- TensorFlow1.2.0
- numpy
- scipy
- sklearn
- matplotlib
- wfdb2.2.0
- pywt

#### Instructions
Currently I will not upload the data-set and label-set due to large size, but you can use wfdb to produce both easily.   

# TODO
- More detail about AD.
- More arrhythmias to be detected.

