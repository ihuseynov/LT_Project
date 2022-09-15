# LT_project

This repository is for LT Project. I have used Glove dataset for pre embedding. This Glove 6B can be found in https://www.kaggle.com/datasets/anindya2906/glove6b and should be added to data folder.


1. This project is designated for papers implemeting Bi-LSTM CRF combination to solve NLP problems such as sequence tagging, NER and etc. 

The papers inspired from:
https://ui.adsabs.harvard.edu/abs/2015arXiv150801991H/abstract
https://www.researchgate.net/publication/333384813_Bidirectional_LSTM-CRF_for_Named_Entity_Recognition


In this project I have tried to address this approach to solve Named Entity Recognition. 

2. To run the project one should run the train.py file. This will generate model and more files within it. After running and training the model, evaulation.py file can be run and you should receive the results on the test data. 

3. The result in this project was 91% F1.
