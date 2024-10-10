''' 
 Title: OrdLR.py
 Description: A python program that implements ordinal regression on datasets containing the results of Static Application Security Testing (SAST) on software. The regression focuses on 
 predicting the level of risk, from none, low, medium, or high. The input columns and test/training split can be selected by the user, and results saved to the desired file, .csv files recommended.
 The program can then output accuracy, weighted F1 score, Mean Absolute Error, and a confusion matrix for the user to determine the results of the training.

 Date: Aug. 26, 2024
 Author: Braiden Little
 Version: 1.0
 Copyright: 2024 Braiden Little

 Acknowledgement: Some of the following code is based on code from the mord ordinal tutorial by Joyita Bhattacharya at https://bjoyita.github.io/Tut11_OrdReg.html

 Program Purpose: 
 To implement an ordinal regression model training in python. This ordinal regression is designed to be used to predict the risk level of software based on results from SAST tools on said software. 
 This program is designed to be used by calling the build_model() and display_results() functions, which in turn call the relevant functions.
 

 Execution: (In windows command line(cmd), assuming the program is in the current directory)
 py main.py
 (this program is designed to be operated with the inlcuded main.py file)
 
 Program Instructions:
 1. Call the program by running the main.py python file.
 
 2. You will be prompted to select a file for training. Select a .csv file as this is the currently supported format by this program. Additionally, the .csv file will require a "risk_level column" for this program.
 So the file may have to be modified to translate an existing "@severity", "Severity", "risk", etc. into a risk_level column. Additionally, the entries in the risk_level column should only contain the values "Low",
 ,"Medium", "High", or "Critical". The files I will test with and include will already have this column made. Occasionally the file selection window will not always be brought up in focus, so keep this in mind as 
 it just may be hidden behind your current window.
 
 3. You will then be prompted to select which other columns in the .csv file will be included for training. You will be given a numbered list of these columns, which you will then enter the corresponding number
 and then enter to include the respective column. At least one column will need to be selected at this point. 
 
 4. You will then be prompted for the test split of the data for training. Enter a decimal value between 0 and 1, exclusive, as the percentage for the training value e.g. 0.8 for an 80% test split.
 
 5. You will then be prompted if you wish for the results to be reproducable. Inputting 'y' here will ensure the data is shuffled the same way every time, and is useful for testing purposes. Whereas any other input
 will cause the data to be shuffled in a different manner, and is more useful for practical training.
 
 6. A file selection window will be selected to save the output file to. If no file is selected, then no output file is saved. The output will be the  combination of the test split of the selected inputs, test outputs, 
 and the predicted values of the model into a .csv file. Occasionally the file save window will not always be brought up in focus, so keep this in mind as it just may be hidden behind your current window. 
 
 
 Program Output:
 After the file is save, the Accuracy, Weighted F1 Score, Mean Absolute Error, and a Confusion Map will be output for the user.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
import time
from tkinter import filedialog
from mord import OrdinalRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, f1_score

class Ordinal_Regression():
    def __init__(self):
        self.model = OrdinalRidge()
        self.label_encoder = LabelEncoder()

    #Prompts the user to select a csv file for training and testing an ordinal regression model
    def get_data(self):
        while True:
            print("Select the .csv file you wish to build the model with.\n")
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename()
            if filename.endswith('.csv'):
                break
            else:
                print("Select a .csv file")
        self.dataset = pd.read_csv(filename)
        #Set dependent and independent variables in their own data frames.
        self.x = self.dataset.loc[:, self.dataset.columns != "risk_level"]
        self.y = self.dataset.loc[:, self.dataset.columns == "risk_level"]
        include_columns = []
        poss_selec = {index + 1: column for index, column in enumerate(list(self.x.columns))}
        #User selection of input variables.
        while True:
            print("Select the number of the of the columns to include as training input, hit Enter with no input to finish.\n")
            #To make the output more readable.
            for k, v in poss_selec.items():
                print(k, v)
            raw_input = input()
            if raw_input == '':
                if len(include_columns) < 1:
                    print('Please select at least one column for training.')
                    time.sleep(1.5)
                else:
                    break
            try: #Check if input is a valid int, if it is append the selection to the included columns.
                selection = int(raw_input)
                include_columns.append(poss_selec[selection])
                del poss_selec[selection]
            except:
                print("Please enter integer input only.\n")
            print("Currently included columns:")
            print(include_columns)
        self.x = self.x[include_columns]

    '''Encode the predictor and independent variables for training and testing
    Also returns the encoders and ordinal_encoder variables for decoding in a separate function.'''
    def encode_data(self):
        #Use Pandas dummies for one hot encoding.
        self.x_encoded = pd.get_dummies(self.x, columns= self.x.columns)
        #Replace any blank fields with "None" for encoding
        self.y = self.y.fillna("None")
        #Encode the output values 
        self.ordinal_encoder = OrdinalEncoder(categories=[["None", "Low", "Medium", "High", "Critical"]])
        self.y_encoded = self.ordinal_encoder.fit_transform(self.y.values.reshape(-1, 1)).ravel()

    #Split the encoded data into test and training sets, with the ratio determined by the user.
    def split_data(self):
        while True:
            split_as_string= input('Enter the test split of the data as a decimal (i.e. 0.8 for 80%)')
            try:
                test_size = float(split_as_string)
                # Can only accept valid percentages
                if test_size >= 1 or test_size <= 0:
                    print('Please enter a decimal number between 0 and 1, such as 0.8.')
                else:
                    break
            except:
                print('Please enter a valid float input, such as 0.8')
        replicate_result = input('Do you wish for the results to be reproducable? y/n')
        replicate_result.lower()
        if replicate_result == 'y':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_encoded, self.y_encoded, test_size = test_size, random_state = 42)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_encoded, self.y_encoded, test_size = test_size)

    #Create and fit the model.
    def create_Model(self):
        self.model.fit(self.x_train, self.y_train)

    #Uses a created model to predict risk levels 
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    #Decode the variables for data presentation and saving.
    def decode_data(self):
        self.y_pred_decoded = self.ordinal_encoder.inverse_transform(self.y_pred.reshape(-1, 1))
        #Add column name, then decode
        self.y_pred_decoded = pd.DataFrame(self.y_pred_decoded, columns = ['Risk_Pred'])
        self.y_test_decoded = self.ordinal_encoder.inverse_transform(self.y_test.reshape(-1, 1))
        #Add column name, then decode
        self.y_test_decoded = pd.DataFrame(self.y_test_decoded, columns = ['Risk_True'])
        self.x_decoded = pd.DataFrame()
        # Identify original x columns by splitting column names from one hot encoding 
        for prefix in set(column.split('_')[0] for column in self.x_test.columns):
            # Filter columns that belong to the same category
            columns = [column for column in self.x_test.columns if column.startswith(prefix)]
            #Identify columns which contain 1, which represents the original category
            self.x_decoded[prefix] = self.x_test[columns].idxmax(axis=1).apply(lambda x: x.split('_')[1])

    #Save the data in an output file
    def save_results(self):
        print("Select an output file to save to.")
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.asksaveasfilename(
           defaultext = '.csv',
           filetypes = [('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if filename:
            # Ensure indices align properly
            self.x_decoded.reset_index(drop=True, inplace=True)
            self.y_test_decoded.reset_index(drop=True, inplace=True)
            self.y_pred_decoded.reset_index(drop=True, inplace=True)
            result_dataset = self.x_decoded.join(self.y_test_decoded).join(self.y_pred_decoded)
            # Save the result to a CSV file
            result_dataset.to_csv(filename)
            print('File ' + filename + ' saved.\n')
        else:
            print('File did not save.')

    #Output the accuracy results 
    def output_accuracy(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print('Accuracy: ' + str(accuracy))

    #Output the mean absolute error result
    def output_mae(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        print('Mean Absolute Error: ' + str(mae))
    
    #Output the F1 Score
    def output_f1_score(self):
        f1 = f1_score(self.y_test, self.y_pred, average= 'weighted')
        print('Weighted F1 Score: ' + str(f1))

    #Display Confusion Matrix
    def output_cm(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm / np.sum(cm), annot= True, fmt='0.2%', cmap= 'GnBu')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def build_model(self):
        self.get_data()
        self.encode_data()
        self.split_data()
        self.create_Model()
        self.predict()
        self.decode_data()
        self.save_results()
    
    def display_results(self):
        self.output_accuracy()
        self.output_f1_score()
        self.output_mae()
        self.output_cm()