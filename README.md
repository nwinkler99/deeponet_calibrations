# About

This repo was used in my Master thesis **Neural Operator Approaches to Deep Calibration of (Rough) Volatility Models** (MS Quant Finance at UZH and ETH).
I have built on the following repositories (see respective copyright licenses):

- https://github.com/amuguruza/NN-StochVol-Calibrations : general inspiration for model setup and analysis plots, as well as validation of my simulation results using TrainrBergomiTermStructure.txt
- https://github.com/ryanmccrickerd/rough_bergomi : utils.py and rbergomi.py functions for simulating the Hybrid scheme
- https://github.com/scipy : optimization functions
- https://github.com/quantlib : Heston pricing functions
- https://github.com/pytorch/pytorch : Deep Learning model setup


Set up a python 3.13.3 env using the provided requirements.txt.
Use training/training.ipynb for training of the models and analysis regarding accuracy, speed etc.
Use real_calibration/ivs_calibration.ipynb for calibrating real iv surfaces and analysis of calibrated parameters over time.
