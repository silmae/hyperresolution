# Extending hyperspectral image wavelength range using a point spectrometer measurement

This repository focuses on fusing together data from a hyperspectral imager (HSI)
and a point spectrometer. The instruments point in the same direction, but the
wavelength ranges are different with only a slight possible overlap: the 
HSI measures shorter wavelengths, and the point spectrometer longer wavelengths.
The goal of this project is to produce a hyperspectral image covering the 
total combined wavelength range of the two instruments. 

The motivation for this comes from properties of the ASPECT imaging module
of the Milani CubeSat, set to fly with the ESA Hera mission to the asteroid
Didymos. ASPECT consists of four instruments, with three hyperspectral
cameras for shorter wavelengths up to 1600 nm and a point spectrometer 
from 1600 nm to 2500 nm. More information of Hera, Milani, and ASPECT can be 
found in this artice: P. Michel et al., *"The ESA Hera Mission: Detailed Characterization 
of the DART Impact Outcome and of the Binary Asteroid (65803) Didymos"*, The Planetary 
Science Journal (2022) (https://doi.org/10.3847/PSJ/ac6f52).

Our approach to processing the data is inspired by a convolutional autoencoder 
architecture originally designed for blind unmixing, presented in this paper: 
B. Palsson, M. O. Ulfarsson and J. R. Sveinsson, *"Convolutional Autoencoder for 
Spectral–Spatial Hyperspectral Unmixing,"*, IEEE Transactions on Geoscience and 
Remote Sensing (2021) (https://doi.org/10.1109/TGRS.2020.2992743). 

The implementation is loosely based on a tutorial that can be found here: 
https://github.com/BehnoodRasti/Unmixing_Tutorial_IEEE_IADF.
Unlike in the tutorial, our network is built with PyTorch.

Currently, the network can somewhat successfully extend the wavelength range 
of some remotely sensed test images. Next up is testing with targets more closely
resembling Didymos. 
