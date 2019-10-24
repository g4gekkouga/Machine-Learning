
# Roll No : 17CS30025
# Name : Pericherla Amshumaan Varma
# Assignment No : 2
# Pandas, Numpy Libraries have been used

import numpy as np
import pandas as pd

# To get the number of instances of different classes
def get_class_dic(Y):
    
    dic = {}
    values = Y.unique()
    
    for val in values:
        dic[val] = Y.value_counts()[val]
    
    return dic

# Get number of instances of differnet values of all attributes in each class
def get_attributes_dic(X, Y):
    
    dic = {}
    values = Y.unique()
    
    for val in values:
        dic[val] = {}
    
    attributes = X.keys()
    
    for att in attributes :
        for val in values:
            dic[val][att] = {}
        
        att_values = X[att].unique()
        
        for att_val in att_values :
            for val in values:
                dic[val][att][att_val] = 0
    
    for index, row, in X.iterrows() :
        for att in attributes :
            dic[Y[index]][att][row[att]] = dic[Y[index]][att][row[att]] + 1
    
    return dic
    
    


# Get conditional probability of each attribute value using laplacian smoothening formula
def laplace_prob(X, Y, a_dic, c_dic):
    
    values = Y.unique()
    attributes = X.keys()
    
    for val in values:
        for att in attributes:
            keys = a_dic[val][att].keys()
            
            # Adding the missing attribute values which do not occur even once
            for i in range(1, 6):
                if i in keys :
                    continue
                else :
                    a_dic[val][att][i] = 0
                    
    for val in values:
        for att in attributes:
            keys = a_dic[val][att].keys()
            
            # laplacian formula  =  Nic + 1 / Nc + no.of.classes
            for i in keys:
                a_dic[val][att][i] = (a_dic[val][att][i] + 1) / (c_dic[val] + len(values))
    
    return a_dic
    

# Predict the class of a test case using the generated probabilities
def get_predict(X, a_dic, c_dic):
    
    values = a_dic.keys()
    attributes = X.keys()
    
    pred = pd.DataFrame(columns=['D', '0', '1']) 
    
    for index, row, in X_test.iterrows() :
        total = 0
        prob = {}
        
        for val in values:
            total = total + c_dic[val]
            
        for val in values:     
            prob[val] = (c_dic[val]) / (total)
        
        for att in attributes :
            for val in values:     
                prob[val] = prob[val] * a_dic[val][att][row[att]]
            
        # Normalizing the class probabilities
        temp = 0
        
        for val in values:
            temp = temp + prob[val]
        
        for val in values:
            prob[val] = prob[val] / temp
        
        max_class = int(max(prob, key=prob.get))
        
        pred = pred.append({'D': max_class, '0': prob[0], '1':prob[1]}, ignore_index=True)
    
    return pred
    
# Get the accuracy of the predicted class values
def test_accuracy(Y, P):
    
    acc = 0
    
    for i in range(len(Y)):
        if Y[i] == P[i] :
            acc = acc + 1
            
    acc = acc * 100 / len(Y)
    
    return acc 
    
# to format and read the data
def get_data(filename):
	
	full = []
	
	with open(filename, 'r') as file:
	    data = file.readlines()

	for i in range(len(data)):
	    data[i] = data[i].replace("\"","")
	    full.append(data[i].strip('\n').split(","))

	data = pd.DataFrame(data=full[1:], columns=full[0])
	data = data.astype(int)
	
	return data


# Reading the data and splitting them into class and attributes dataframes
X_data1 = get_data('data2_19.csv')
X_data2 = get_data('test2_19.csv')

X_train = X_data1.iloc[:,1:].reset_index(drop=True)
Y_train = X_data1.iloc[:,0].reset_index(drop=True)

X_test = X_data2.iloc[:,1:].reset_index(drop=True)
Y_test = X_data2.iloc[:,0].reset_index(drop=True)

# Get the required model summaries using above defined functions

class_dic = get_class_dic(Y_train)

att_dic = get_attributes_dic(X_train, Y_train)

att_dic = laplace_prob(X_train, Y_train, att_dic, class_dic)

predictions_proba = get_predict(X_test, att_dic, class_dic)


# Printing the results
print()
print("Actual Classes of Test Cases : ")
print(Y_test, end='\n')

print()
print("Predicted Classes with Prediction Probabilities (Normalized) of each Class : ")
print(predictions_proba, end='\n')



predictions = predictions_proba.iloc[:,0]
accuracy = test_accuracy(Y_test, predictions)
print()
print("Total Accuracy of the Naive Bayes Classifier on the given Test Cases is : ")
print(accuracy,'%', end='\n')
print()

