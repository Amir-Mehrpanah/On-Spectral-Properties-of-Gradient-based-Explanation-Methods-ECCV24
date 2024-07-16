# On Spectral Properties of Gradient-based Explanation Methods (published in ECCV24)

This repository is used to reproduce all experiments discussed in our paper.

We are in the process of refactoring this code for our next work. But essentially the code is simple, you first run a python script in commands folder, this automates a series of python codes that need to run for one experiment. Each code in the command folder then calls driver.py and this starts calling other functions in the code which should be easy to follow or edit. The result of driver will be in two folders called raw data (actual results of a computation) and metadata that contains experiment variables in a pandas compatible way. So it'll make it easy to select experiments with certain criteria. Then you need experiments folder to acutally interactively go over each experiments and handle visualizations.

Please refer to the [paper](#) for more details.
