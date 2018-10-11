# Generic-word-and-name-predictor
I build a small tool which can distinguish between generic words and names using an artificial neural net.

## Problem
The problem statement was to develop a (machine learning) tool capable of predicting whether a name is generic or the name of a person. The program will receive input in the form of a list of names and surnames and must predict whether both and generic or not and make an overall assignment if the names are generic or not. The training data is a very limited list of names, and even more limited list of non-names. 

## Approach
Since a list of names were provided, this provided an opportunity to first run all unclassified words through this list and classify as either of the labels in the training set. If the word is not contained within this set, a machine learning approach will be taken. 
The following set of features were extracted from the training set, which aided in the classification of words into the two groups. These features are commonly thought to be either characteristic of names or generic words: Length of the word, Number of spaces in the word, Presence and amount of numbers, Presence and number of capitalized letters, Percentage of vowels in the word, Punctuation in the word, Presence of two or more consecutive capital letters, Number of syllables and the readability of the word. These features had the highest predictive power, but I also included a bag of words approach for counting the combinations of letters in a word. E.g., Count will be divided into ‘co’, ‘ou’, ‘un’, ‘nt’ and have a count of 1 for each. This also increased the predictive power of the model.
I decided to use an artificial neural network to classify the words as generic or not. This is due to the large number of features generated, the imbalance in the dataset and the possible non-linearity in the relationship between word features and identity. The network was constructed in Keras, using a Tensorflow backend. The network had an input layer, three hidden layers (sizes 64, 8, 16) and one output layer. The model was evaluated using accuracy, precision and recall. It achieved an unseen accuracy of more than 98.5% with most errors being false positives (generic names being classified as true names). This however can be overcome by changing the acceptance parameter, trading in some of the recall (more false negatives) for an increase in precision (less false positives). This will largely depend on the desired outcome of the application.  

## Usage
Run the `generic_name_predictor.py` script with the `final_model.h5` file needed for Keras. 
You will be prompted to provide a test case file which has one column containing names and one column containing surnames.
This all in a Graphical User Interface.
