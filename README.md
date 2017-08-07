# Variational Autoencoder for predicting mutational signatures
This work is a course project for CSC2541.

### Background
Cell processes leave a unique signature of mutation types in cancer genome. Using the mutational signatures, it is possible to infer the fraction of mutations contributed by each cell process. Mutational signatures are represented as multinomial distributions over 96 mutation types. Using our framework Trackature (https://bitbucket.org/Erise/trackature), we can infer mutational signatures changing over time.

### Model
Input: through Trackature we obtain a time series of mutation distributions (represented as multinomials) and signature exposures. Given the data from the previous time point, I want to predict the data (mutation distribution and exposures) at the current time point using variational autoencoder.

### Code
I use the implementation of variational autoencoder from HIPS/autograd. My model is implemented in selected files in examples/ directory. This is a preliminary work and is not intended for public use.

### Dependency 
Autograd python package
