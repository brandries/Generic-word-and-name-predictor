#!C:\\ProgramData\\Anaconda3\\python.exe
# We want to first train a model on the names given in the example dataset.
# As input to the model we use engineered features found the the TextProcessing
# module
# We use the Keras framework to build a simple Artificial Neural Network.

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from textstat.textstat import textstat
from TextProcessing import *
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model

# read data in
def read_data():
    print('Reading in the training set \n ----------------------------')
    train = pd.read_excel('A_training_data.xlsx')
    return train


# Functions to extract features from the data

def remove_trail_ws(word):
    return word.strip()

# List of functions to do
def make_features(train):
    functions = {'wordlength': WordLength(),
                'hasspaces': HasSpaces(),
                'hasnumber': HasNumbers(),
                'isupper': HasUppers(),
                'numbers': NumberOfNumbers(),
                'uppers': NumberOfUppers(),
                'vowels': Vowels(),
                'punctuation': Punctuation(),
                'nocap_space': MoreCapitals(),
                'syllables': Syllables(),
                'readable': Readability()
                }
    print('Creating the features \n  ----------------------------')
    train['name'] = train['name'].apply(remove_trail_ws)
    for i, j in functions.items():
        train[i] = train['name'].apply(lambda x: j.transform(x))

    cv_ngrams = TfidfVectorizer(ngram_range=(2, 7), analyzer='char', min_df=0.001) # Create bag of words using n-grams 2-7
    ngrams = cv_ngrams.fit_transform(train['name'])

    X = train.drop(['name',
                    'Display Name'],
                axis=1).join(pd.DataFrame(ngrams.toarray(),
                                            columns=cv_ngrams.vocabulary_),
                                rsuffix='_ngram')
    y = train['Display Name']

    return X, y

def train_test(X, y):
    print('\nAre we doing the final round of training? [Y/N]')
    final_train = input()
    if final_train in ['Yes', 'yes', 'y', 'Y']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test

# Create neural network in keras to classify names not in the list
# Function to build the model

def build_model(layer1, layer2, layer3, act1, act2, act3, lr, X_train):
    model = Sequential()
    model.add(Dense(layer1, activation=act1, input_shape=(X_train.shape[1], )))
    model.add(Dense(layer2, activation=act2))
    model.add(Dense(layer3, activation=act3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(lr=lr), loss='binary_crossentropy', metrics=["accuracy"])
    return model

def run_model(X_train, y_train):
    K.clear_session()
    earlystop = EarlyStopping(patience=10)
    model = build_model(64, 8, 16, 'tanh', 'relu', 'relu', 0.0001, X_train)
    model.summary()
    print('Fitting model  \n ----------------------')
    model.fit(X_train, y_train, epochs=200,
            validation_split=0.1, callbacks=[earlystop], verbose=1)
    return model

def save_model(model):
    print('\nWhat would you like to save the model as?')
    model_name = input()
    model.save(model_name + 'h5')

# Make predictions
def evaluate_model(train, X_test, y_test, model):
    print('----------------------\nEvaluating the model\n----------------------')
    pred_proba = pd.DataFrame(model.predict(X_test),
                            columns=['proba'], index=y_test.index)
    pred_proba['class'] = [1 if x >= 0.4 else 0 for x in pred_proba['proba']]
    print(metrics.classification_report(y_test, pred_proba['class']))
    print(metrics.confusion_matrix(y_test, pred_proba['class']))
    whichwords = train.join(pred_proba).dropna()[['name', 'Display Name', 'class']]
    whichwords.head()
    subframe = whichwords[whichwords['Display Name'] != whichwords['class']]
    print('There were {} words incorrectly classified'.format(len(subframe)))
    print('Words not classified\n{}'.format(subframe.sort_values(['Display Name'])))

def main():
    train = read_data()
    X, y = make_features(train)
    X_train, X_test, y_train, y_test = train_test(X, y)
    model = run_model(X_train, y_train)
    save_model(model)
    evaluate_model(train, X_test, y_test, model)

if __name__ == '__main__':
    main()