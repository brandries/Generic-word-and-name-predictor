# The Main program to predict if words are generic or not
# This program combined sorting words using a predefined list
# As well as using a machine learning approach to sort names

import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from TextProcessing import *
from keras.models import load_model

LARGE_FONT = ('Verdana', 12)
MEDIUM_FONT = ('Helvetica', 10)
SMALL_FONT = ('Helvetica', 8)

f = Figure()
a = f.add_subplot(111)

# read data in
def read_data(name):
    print('Reading in the training set \n ----------------------------')
    train = pd.read_excel(name)
    return train

# make list of generic and names
def names_list(train):
    names = []
    non_names = []
    for i, j in zip(train['name'], train['Display Name']):
        if j == 0:
            names.append(i)
        elif j == 1:
            non_names.append(i)
        else:
            print(i)
    print('{} names in the train set and {} non-names \n'.format(len(names),
                                                                 len(non_names)))
    return names, non_names
# Functions to extract features from the data

def remove_trail_ws(word):
    return word.strip()

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
    cv_ngrams.fit(train['name'])
    return train, cv_ngrams, functions

# Put everything together
# Function to assign name as generic or not


def generic_name(word, names, non_names, functions_dict,
                 ngram_vectorizer, model, threshold):
    if word in names:
        return 0
    elif word in non_names:
        return 1
    else:
        # ML part to predict if not in reference set
        df = pd.DataFrame([word], columns=['name'])
        for i, j in functions_dict.items():
            df[i] = df['name'].apply(lambda x: j.transform(x))
        ngram = ngram_vectorizer.transform(df['name'])
        X = df.drop(['name'],
                    axis=1).join(pd.DataFrame(ngram.toarray(),
                                              columns=ngram_vectorizer.vocabulary_),
                                 rsuffix='_ngram')
        pred = model.predict_proba(X)
        if pred >= threshold:
            return 1
        else:
            return 0

# Check if both columns are generic or not

def double_pass(name, surname):
    if name == 0 and surname == 0:
        return 0
    else:
        return 1

def save_output_nowindow(test, outname):   
    test.to_csv(outname, index=False)

def save_file(file):
    f = filedialog.asksaveasfilename(defaultextension=".csv")
    if f is None:
        return
    file.to_csv(f, index=False)
    

def popupmesg(msg):
    popup = tk.Tk()
    popup.wm_title('!')
    label = ttk.Label(popup, text=msg, font=MEDIUM_FONT)
    label.pack(side='top', fill='x', pady=10)
    b1 = ttk.Button(popup, text='Okay', command = popup.destroy)
    b1.pack()
    popup.geometry('600x170')
    popup.mainloop()


def animate(i):
    test = pd.read_csv('Output_from_model.csv')
    a.clear()
    a.table(cellText=test.values,colWidths = [0.2]*len(test.columns),
              rowLabels=test.index, 
              colLabels=test.columns,
              cellLoc = 'center', rowLoc = 'center',
              loc='center')
    a.xaxis.set_visible(False) 
    a.yaxis.set_visible(False)
    a.axis("off")
    a.axis('tight')



class MakeMainGUI(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default='icon.ico')
        tk.Tk.wm_title(self, 'Classify words as Name or Generic')
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Predict Model',
                            command=lambda:self.show_frame(PredictPage))
        filemenu.add_command(label='Show Results',
                            command=lambda:self.show_frame(PlotPage))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=quit)
        menubar.add_cascade(label='File', menu=filemenu)

        buildmenu = tk.Menu(menubar, tearoff=0)
        buildmenu.add_command(label='Load Dataset',
                               command=lambda: self.load_file())

        menubar.add_cascade(label='Model Parameters', menu=buildmenu)


        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        # Add pages to this loop to have it in frames
        for F in (StartPage, PredictPage, PlotPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()
        
    def load_file(self):
        self.fileName = filedialog.askopenfilename(filetypes = (("CSV files", "*.csv")
                                                     ,("Excel files", "*.xlsx;*.xls")
                                                     ,("All files", "*.*") ))
        if self.fileName[-3:] == 'lsx':
            with open(self.fileName, 'rb') as input_file:
                self.test = pd.read_excel(input_file)
        elif self.fileName[-3:] == 'csv':
            with open(self.fileName, 'r') as input_file:
                self.test = pd.read_csv(input_file)


class StartPage(tk.Frame):

    def __init__(self, parent, controler):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='''This program is designed to classify names as being 
        generic or not. The program relies on name first 
        being present in a sample list, which is searched 
        but then uses Deep Learning to classify names which 
        are not present in the list.''', font=MEDIUM_FONT)
        label.pack(padx=10, pady=10)
        button2 = ttk.Button(self, text='Predict Page',
                            command=lambda:controler.show_frame(PredictPage))
        button2.pack()
        button3 = ttk.Button(self, text='Output Page',
                            command=lambda:controler.show_frame(PlotPage))
        button3.pack()



class PredictPage(tk.Frame):

    def __init__(self, parent, controler):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Train the model', font=LARGE_FONT)
        label.pack(padx=10, pady=10)
        button1 = ttk.Button(self, text='Back to Home',
                            command=lambda:controler.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text='Load in the test Data',
                             command=lambda:controler.load_file())
        button2.pack()
        button3 = ttk.Button(self, text='Do the predictions for loaded file',
                             command=lambda:self.main(controler))
        button3.pack()
        button4 = ttk.Button(self, text='Save File',
                             command=lambda:save_file(controler.test))
        button4.pack()
        button5 = ttk.Button(self, text='Plot output table',
                             command=lambda:controler.show_frame(PlotPage))
        button5.pack()


    def main(self, controler):
        train = read_data('A_training_data.xlsx')
        names, non_names = names_list(train)
        train, cv_ngrams, functions = make_features(train)
        model = load_model('final_model.h5')
        try:
            controler.test['Is_name_generic'] = controler.test.iloc[:, 0].apply(generic_name, args=(names,
                                                                                non_names,
                                                                                functions,
                                                                                cv_ngrams,
                                                                                model,
                                                                                0.4))
            controler.test['Is_surname_generic'] = controler.test.iloc[:, 1].apply(generic_name,
                                                            args=(names,
                                                                    non_names,
                                                                    functions,
                                                                    cv_ngrams,
                                                                    model,
                                                                    0.4))
            controler.test['Is_record_generic'] = controler.test.apply(lambda x: double_pass(x['Is_name_generic'],
                                                                        x['Is_surname_generic']),
                                                axis=1)
            save_output_nowindow(controler.test, 'Output_from_model.csv')
            popupmesg('Done predicting, please specify a file name to save the file as and view the table')
            
        except:
            popupmesg('''   
            Please load in the test data first.

            The test data should contain two columns, "Name" and "Surname."
            The test data can be in csv or Excel file format.''')

        

class PlotPage(tk.Frame):
 
    def __init__(self, parent, controler):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Prediction Page', font=LARGE_FONT)
        label.pack(padx=10, pady=10)
        button1 = ttk.Button(self, text='Back to Home',
                            command=lambda:controler.show_frame(StartPage))
        button1.pack()

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side = tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side = tk.TOP, fill=tk.BOTH, expand=True)



app = MakeMainGUI()
app.geometry('1280x720')
ani = animation.FuncAnimation(f, animate, interval=5000)
app.mainloop()

