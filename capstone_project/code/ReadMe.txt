Training a model for Text recognition using Convolution+Recurrent Neural Networks (CRNN)
Dataset is obtained from https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth
Dataset contains Images with text and name of the Image contains the text present inside the image
Dataset is split into Train and Validation sets. 
Train-180000
Valid-20,000
Model is trained for 50 epochs. 
Analysis during Evaluation Time. 
The model is evaluated on the eval set which is not used for training and below observations are obtained.
	1. Model is doing good even on images having text with very less visibility.
	2. Model has recorded incorrect predictions for P and D and sometimes unable to recognize R word.
	3. This trained model is capable of detecting characters/Numbers of english which are written in any font.

References: https://arxiv.org/pdf/1507.05717.pdf
2. https://theailearner.com/