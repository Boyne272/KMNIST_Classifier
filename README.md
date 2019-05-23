# ACSE-8-Mini-Project

## Repository Structure
- spec/ - the project decription files as well as some admin bits
- data/ - the given Kuzushiji-MNIST data files
- models/ - contains all the models we trained, there predictions and there comparisons. Naming conventions are a bit all-over the place but this is just a working space
- training/ - notebooks which define and train all the different models we tested. Is quite a messy but is mostly legacy work with many of the networks not performing well enough to be submitted
- tools.py - methods for training a neural network, inspecing it as well as a few other commonly used functions such as setting the seeds or creating the output csv files wanted
- AlexNet5.ipynb, AlexNet7.ipynb, Ensemble.ipynb - notebooks used to train and combine the final submission network ensemble (see notes below)
- Ensemble_2.csv, Ensemble_4.csv - final test submissions (submitted under the names "AlexNeyt7_combo (1).csv" and "ensemble (1).csv" respectively)

## Notes to the Examiners
Our submissions were both formed of model ensembles, namely combinations of AlexNet7 and AlexNet5 networks each trained with two different optimisers (Adam and SGD). Either all 4 were combined (Ensemble_4) or just the AlexNet7 networks (Ensemble_2). The notebooks show how the data was loaded, pre-processed and augmented as well as the structures of each network and how they were trained (with the use of tools.py). The Ensemble notebook shows how the models were combined to create our final output.

Over all we enjoyed the challenge though we possibly worked a little too hard on it and were gutted by coming second by such a close margin. Here is our last appropirate meme:

<a href="https://imgflip.com/i/31qum1"><img src="https://i.imgflip.com/31qum1.jpg" title="table flip meme" alt="table flip mem"/></a>

## Important Links:

- Team Registration

https://www.kaggle.com/t/3713b8edcaab4ac7ac6045d7353c1aba

- Kaggle Competition Site + Leaderboard

https://www.kaggle.com/c/acse-module-8-19/overview
