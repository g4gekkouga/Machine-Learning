# Roll No : 17CS30025
# Name : Pericherla Amshumaan Varma
# Assignment No : 3
# Pandas, Numpy and Math lib have been used

# pandas library has been used just as a datatype for input data. No functions in pandas that build the tree directly have been used

import numpy as np
import pandas as pd
import math
import random
# import pickle


def entropy(Y):  

    Ent = 0
    values = Y.unique()
    for val in values:
        frac = Y.value_counts()[val] / len(Y)
        Ent = Ent - frac * math.log(frac)
    return Ent



def gain(X, Y, att):

	Ent_X = entropy(Y)
	values = X[att].unique()
	Ent_sum = 0
	for val in values:
		index = X.index[X[att] == val].tolist()
		Y_temp = Y.iloc[index]
		Y_temp = Y_temp.reset_index(drop=True)
		frac = len(Y_temp)/len(Y)
		Ent_sum = Ent_sum + frac * entropy(Y_temp)
	return (Ent_X - Ent_sum)




def decide_att(X, Y):

	attribute = X.keys()[0]
	_gain = 0
	for att in X.keys():
		temp = gain(X, Y, att)
		if temp > _gain:
			_gain = temp
			attribute = att
	return attribute



def get_sub_data(X, Y, att, val):

	index = X.index[X[att] == val].tolist()
	X_temp = X.iloc[index, : ]
	Y_temp = Y.iloc[index]
	X_temp = X_temp.reset_index(drop=True)
	Y_temp = Y_temp.reset_index(drop=True)
	return X_temp, Y_temp



def get_tree(X, Y, count, p_att, tree=None):
	att = decide_att(X, Y)
	values = X[att].unique()
	if tree is None:                    
		tree = {}
		tree[att] = {}
	
	for val in values:
		X_sub, Y_sub = get_sub_data(X, Y, att, val)
		y_values = Y_sub.unique()
		yes = 0
		no = 0
		for y_val in y_values:
			if y_val == 'yes':
				yes = Y_sub.value_counts()['yes']
			if y_val == 'no':
				no = Y_sub.value_counts()['no']
		
		if no == 0:
			ratio = 150
		else :
			ratio = float(yes / no)

		if att == p_att:
			if yes > no:
				return 'yes'
			else:
				return 'no'
            
		elif count > 1: # As only 3 attributes in this case
			if yes > no:
				tree[att][val] = 'yes'
			else:
				tree[att][val] = 'no'
		
		else :
			tree[att][val] = get_tree(X_sub, Y_sub, count+1, att)
		
	return tree




def print_tree(dic,level):
    if type(dic)!=dict:
        print(": "+dic)
        return
    for key in dic:
        print()
        val = dic[key]
        if type(val)==dict:
            for k in val:
            	for i in range(level-1):
                    print("\t",end="")
            	if level!=0 :
            		print("|---->", end="")
            		print("\t",end="")
            	print(key+" = "+str(k),end=" ")
            	print_tree(val[k],level+1)



X_data = pd.read_csv('data3_19.csv')
X_test_data = pd.read_csv('test3_19.csv', sep=',', names=["pclass", "age", "gender", "survived"])
X_train = X_data.iloc[:,:-1].reset_index(drop=True)
X_test = X_test_data.iloc[:,:-1].reset_index(drop=True)
Y_train = X_data.iloc[:,-1].reset_index(drop=True)
Y_test = X_test_data.iloc[:,-1].reset_index(drop=True)



# Uncomment this section and comment the below one to Split the data into test and training sets 
'''
shuffled_indices=np.random.permutation(len(X_data))
test_set_size= int(len(X_data)*0.2)
test_indices=shuffled_indices[:test_set_size]
train_indices=shuffled_indices[test_set_size:]
X_train = X_data.iloc[train_indices,:-1].reset_index(drop=True)
X_test = X_data.iloc[test_indices,:-1].reset_index(drop=True)
Y_train = X_data.iloc[train_indices,-1].reset_index(drop=True)
Y_test = X_data.iloc[test_indices,-1].reset_index(drop=True)
'''



def get_cumulated_weights(weights):
 #   c_weights = pd.DataFrame(columns=['margin'])
    c_weights = []
    length = len(weights)
 #   c_weights.loc[0] = weights.loc[0]['probability']
    c_weights.append(weights[0])
    for index in range(1, length):
 #       c_weights.loc[index] = c_weights.loc[index-1]['margin'] + weights.loc[index]['probability']
        c_weights.append(c_weights[index-1] + weights[index])
    return c_weights





def get_new_train_set(train_set_x, train_set_y, weights):
    length = len(train_set_x)
    ind_list = []
    for i in range(length):
        ind_list.append(i)
    new_ind = np.random.choice(ind_list, size = length, p = weights)
    '''
    for i in range(length):
        ind = np.random.choice(ind_list, p = weights)
        new_ind.append(ind)
    '''
    #row = train_set_x.loc[ind, :]
    #new_train_set_x = new_train_set_x.append(row, ignore_index=True)
    new_train_set_x = train_set_x.iloc[new_ind].reset_index(drop=True)
    new_train_set_y = train_set_y.iloc[new_ind].reset_index(drop=True)
    return new_train_set_x, new_train_set_y





'''
def get_new_train_set(train_set_x, train_set_y, c_weights):
    length = len(train_set_x)
    new_train_set_x = pd.DataFrame(columns=['pclass', 'age', 'gender'])
    new_train_set_y = pd.DataFrame(columns=['survived'])
    for index in range(length):
        num = random.random()
        ind = 0
        for j in range(length):
            if num > c_weights[j]:
                continue
            ind = j
            break
        row = train_set_x.loc[ind, :]
        new_train_set_x = new_train_set_x.append(row, ignore_index=True)
        new_train_set_y.loc[index] = train_set_y.loc[ind]
    return new_train_set_x, new_train_set_y
'''




def test_validity(tree, train_x, train_y):
	validity = []
	for index, row, in train_x.iterrows():
	    ini = tree.keys()
	    i = list(ini)[0]
	    val1 = row[i]
	    sec = tree[i].keys()
	    for j in sec:
	    	if val1 == j:
	    		thr = tree[i][j]
	    		if (thr == 'yes') | (thr == 'no'):
	    			if train_y[index] == thr:
	    				validity.append(1)
	    				continue
	    			else:
	    				validity.append(0)
	    				continue
	    		thr = thr.keys()
	    		k = list(thr)[0]
	    		val2 = row[k]
	    		fou = tree[i][j][k].keys()
	    		for l in fou:
	    			if val2 == l:
	    				fiv = tree[i][j][k][l]
	    				if (fiv == 'yes') | (fiv == 'no'):
	    					if train_y[index] == fiv:
	    						validity.append(1)
	    						continue
	    					else:
	    						validity.append(0)
	    						continue
	    				fiv = fiv.keys()
	    				m = list(fiv)[0]
	    				val3 = row[m]
	    				six = tree[i][j][k][l][m].keys()
	    				for n in six:
	    					if val3 == n:
		    					sev = tree[i][j][k][l][m][n]
		    					if train_y[index] == sev:
		    						validity.append(1)
		    						continue
		    					else:
		    						validity.append(0)
		    						continue
	for i in range(len(validity), len(train_x)):
		validity.append(0)

	return validity




def get_epsilon(validity, weights):
    epsilon = 0
    for i in range(len(validity)):
        if validity[i] == 0 :
            epsilon = epsilon + weights[i]
    return epsilon



def get_alpha(epsilon):
    if epsilon == 0:
        return 6
    alpha = np.log((1 - epsilon) / epsilon)
    alpha = alpha / 2
    return alpha



def update_weights(validity, weights, alpha):
    new_weights = []
    new_val = 0
    neg_alpha = -1*alpha
    for i in range(len(validity)):
        if validity[i] == 0 :
            new_val = weights[i] * math.exp(alpha)
            new_weights.append(new_val)
        else : 
            new_val = weights[i] * math.exp(neg_alpha)
            new_weights.append(new_val)
    return new_weights            




def normalize_weights(weights):
    total = np.sum(weights)
    normalized_weights = []
    for i in range(len(weights)):
        normalized_weights.append(weights[i] / total)
    return normalized_weights



def get_output(tree, test_x):
	output = []
	for index, row, in test_x.iterrows():
	    ini = tree.keys()
	    i = list(ini)[0]
	    val1 = row[i]
	    sec = tree[i].keys()
	    for j in sec:
	    	if val1 == j:
	    		thr = tree[i][j]
	    		if (thr == 'yes') | (thr == 'no'):
	    			output.append(thr)
	    			continue
	    		thr = thr.keys()
	    		k = list(thr)[0]
	    		val2 = row[k]
	    		fou = tree[i][j][k].keys()
	    		for l in fou:
	    			if val2 == l:
	    				fiv = tree[i][j][k][l]
	    				if (fiv == 'yes') | (fiv == 'no'):
	    					output.append(fiv)
	    					continue
	    				fiv = fiv.keys()
	    				m = list(fiv)[0]
	    				val3 = row[m]
	    				six = tree[i][j][k][l][m].keys()
	    				for n in six:
	    					if val3 == n:
		    					sev = tree[i][j][k][l][m][n]
		    					output.append(sev)
		    					continue
	for i in range(len(output), len(test_x)):
		output.append('yes')

	return output



def get_class(c_outputs, alpha, n):
    mylist = []
    minimum = 100000;
    for i in range(n):
        if minimum > len(c_outputs[i]):
            minimum = len(c_outputs[i])
        
    for i in range(minimum):
        temp_list = [0, 0]
        for j in range(n):
            if c_outputs[j][i] == "yes":
                temp_list[1] += alpha[j]
            else:
                temp_list[0] += alpha[j]
        mylist.append(temp_list)
       # print(temp_list)
    class_output = []
    for i in range(minimum):
        if mylist[i][1] >= mylist[i][0] :
            class_output.append("yes")
        else :
            class_output.append("no")
    return class_output, minimum                      



train_set_x = X_train
train_set_y = Y_train
num_iter = int(input("Enter number of iterations : "))
# weights = pd.DataFrame(columns=['probability'])
weights = []
# print(weights)
total = len(train_set_x)
ratio = 1/total
for index in range(total):
    weights.append(ratio)
#    weights.loc[index]=ratio
# print(len(weights))




weights = normalize_weights(weights)
trees = []
alpha = []
epsilon = []
for i in range(num_iter):
#   cumulated_weights = get_cumulated_weights(weights)
#   print(cumulated_weights)
    new_train_set_x, new_train_set_y = get_new_train_set(train_set_x, train_set_y, weights)
#    print(new_train_set_y[0])
#   print(new_train_set_x)
#   print(new_train_set_y)
#    new_train_set_x = new_train_set_x.reset_index(drop=True)
#    new_train_set_y = new_train_set_y.reset_index(drop=True)
    new_tree = get_tree(new_train_set_x, new_train_set_y, 0, None)
    trees.append(new_tree)
    validity = test_validity(new_tree, new_train_set_x, new_train_set_y)
#    print_tree(new_tree, 0)
#    print(len(validity))
    new_epsilon = get_epsilon(validity, weights)
    epsilon.append(new_epsilon)
#    print(new_epsilon)
    new_alpha = get_alpha(new_epsilon)
    alpha.append(new_alpha)
#    print(new_alpha)
    validity_original = test_validity(new_tree, train_set_x, train_set_y)
    weights = update_weights(validity_original, weights, new_alpha)
#    print(weights)
    weights = normalize_weights(weights)
    


'''

with open("trees_250.txt", "wb") as fp:
    pickle.dump(trees, fp)

with open("alpha_250.txt", "wb") as fp:
    pickle.dump(alpha, fp)
    
with open("epsilon_250.txt", "wb") as fp:
    pickle.dump(epsilon, fp)

'''

'''
for i in range(num_iter):
    print_tree(trees[i], 0)
'''



test_set_x = X_test
test_set_y = Y_test




# single classifier accuracy 
single_tree = get_tree(train_set_x, train_set_y, 0, None)
# print_tree(single_tree, 0)
single_validity = test_validity(single_tree, test_set_x, test_set_y)
# print(single_validity)
correct = single_validity.count(1)
incorrect = single_validity.count(0)
# print(correct, incorrect)
accuracy = 100 * correct / (correct + incorrect)
print("Accuracy by single classifier : ")
print(accuracy, "%")
print(" ")




# Adaboost classifier accuracy
classifier_outputs = []
for i in range(num_iter):
    round_output = get_output(trees[i], test_set_x)
    classifier_outputs.append(round_output)
'''
for i in range(num_iter):
    print(classifier_outputs[i])
'''
final_output, minimum = get_class(classifier_outputs, alpha, num_iter)
#print(final_output)
correct_ = 0
incorrect_ = 0
for i in range(minimum):
    if final_output[i] == test_set_y[i]:
        correct_ += 1
    else:
        incorrect_ += 1
print("Accuracy by Adaboost : ")
print(100 * correct_ / (correct_ + incorrect_), "%")
print(" ")
print("Epsilon values for respuctive rounds : ")
for i in range(num_iter):
	print(epsilon[i])
print("")
print("Alpha values for respuctive rounds : ")
for i in range(num_iter):
	print(alpha[i])




