# About

This repo was used in my Master thesis **Neural Operator Approaches to Deep Calibration of (Rough) Volatility Models** (MS Quant Finance at UZH and ETH). Use the training.ipynb notebook for training and performance evaluation on synthetic data. Use ivs_calibration for calibrating and testing on real option data.

I have built on the following repositories:

- https://github.com/amuguruza/NN-StochVol-Calibrations/tree/master
- https://github.com/svenhsia/Calibrating-Rough-Volatility-Models-with-Deep-Learning/tree/master

Set up a python env using the provided requirements.txt.
Use training/training.ipynb for training of the models and analysis regarding accuracy, speed etc.
Use real_calibration/ivs_calibration.ipynb for calibrating real iv surfaces and analysis of calibrated parameters over time.