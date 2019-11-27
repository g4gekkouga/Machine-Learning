#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


X_data = pd.read_csv('NURSERY.csv', names=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"])


# In[3]:


shuffled_indices=np.random.permutation(len(X_data))
test_set_size= int(len(X_data)*0.33)
test_indices=shuffled_indices[:test_set_size]
train_indices=shuffled_indices[test_set_size:]
X_train = X_data.iloc[train_indices,:-1].reset_index(drop=True)
X_test = X_data.iloc[test_indices,:-1].reset_index(drop=True)
Y_train = X_data.iloc[train_indices,-1].reset_index(drop=True)
Y_test = X_data.iloc[test_indices,-1].reset_index(drop=True)


# In[4]:


'''
X_train = X_data.iloc[:,:-1].reset_index(drop=True)
X_test = X_data.iloc[:,:-1].reset_index(drop=True)
Y_train = X_data.iloc[:,-1].reset_index(drop=True)
Y_test = X_data.iloc[:,-1].reset_index(drop=True)
'''


# In[ ]:





# In[5]:


def entropy(Y):  
    Ent = 0
    values = Y.unique()
    for val in values:
        frac = Y.value_counts()[val] / len(Y)
        Ent = Ent - frac * math.log(frac)
    return Ent


# In[6]:


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


# In[7]:


def decide_att(X, Y, parent_att):
	attribute = None
	_gain = -100000
	for att in X.keys():
		temp = gain(X, Y, att)
		if temp > _gain:
			if (att in parent_att):
				continue
			_gain = temp
			attribute = att
	if attribute is None:
		return parent_att[-1]
	return attribute
    


# In[8]:


def get_sub_data(X, Y, att, val):

	index = X.index[X[att] == val].tolist()
	X_temp = X.iloc[index, : ]
	Y_temp = Y.iloc[index]
	X_temp = X_temp.reset_index(drop=True)
	Y_temp = Y_temp.reset_index(drop=True)
	return X_temp, Y_temp


# In[9]:


def get_tree(X, Y, parent_att, count, tree = None):
	current_att = decide_att(X,Y,parent_att)
	values = X[current_att].unique()
	if tree is None:                    
		tree = {}
		tree[current_att] = {}
	for val in values:
		X_sub, Y_sub = get_sub_data(X, Y, current_att, val)
		y_values = Y_sub.unique()
		class_count = {}
		for y_val in y_values:
			class_count[y_val] = Y_sub.value_counts()[y_val]
		maximum = max(class_count, key=class_count.get)
		total = 0
		for i in class_count.values():
			total = total + i
		if (count <= 1):
			tree[current_att][val] = maximum
		elif(class_count[maximum]/total == 1):
			tree[current_att][val] = maximum
		else:
			new_parents = parent_att.copy()
			new_parents.append(current_att)
			tree[current_att][val] = get_tree(X_sub, Y_sub, new_parents, count-1)
	return tree


# In[10]:


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


# In[11]:


def test_accuracy(ptX, ptY, dic, level):
    if type(dic)!=dict:
        if (dic == ptY):
            return 1
        else:
            return 0
    for key in dic:
        value = ptX[key]
        val = dic[key]
        if type(val)==dict:
            if value in val:
                ret_val = test_accuracy(ptX, ptY, val[value], level+1)
                return ret_val
            else:
                avg = []
                for i in val:
                    return avg.append(test_accuracy(ptX, ptY, val[i], level+1))
                if (avg.count(1) >= avg.count(0)):
                    return 1
                else:
                    return 0


# In[12]:


parents = []
tree = get_tree(X_train, Y_train, parents, 8, None)


# In[13]:


print_tree(tree,0)


# In[14]:


test_pts = X_test.to_dict(orient='records')
print(len(test_pts))


# In[15]:


true = 0
false = 0
for i in range(42):
    pred = test_accuracy(test_pts[i], Y_test[i], tree, 0)
    if (pred == 1): 
        true = true + 1
    else:
        false = false + 1
accuracy = (true*100) / (true + false)
print("Accuracy is ", accuracy)
    


# In[ ]:




