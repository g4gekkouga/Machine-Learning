# Roll No : 17CS30025
# Name : Pericherla Amshumaan Varma
# Assignment No : 4 - Clusters

import numpy as np
import math

# to read data from file
def ReadData(fileName): 
    f = open(fileName, 'r');
    lines = f.read().splitlines() 
    f.close() 
    items = []
    items_value = []
    for i in range(0, len(lines)-1): 
        line = lines[i].split(',')
        itemFeatures = []
        for j in range(len(line)-1): 
            v = float(line[j])
            itemFeatures.append(v)  
        items.append(itemFeatures)
        items_value.append((line[len(line)-1]))
    return items, items_value

# Randomly initialize the cluster means
def InitializeMeans(data, n):
    means = []
    ind_list = []
    for i in range(len(data)):
        ind_list.append(i) 
    indices = np.random.choice(ind_list, size = n)
    for i in range(n):
        means.append(data[indices[i]])
    return means

def eucli_dist(pt1, pt2):
    dist = 0
    for i in range(len(pt1)):
        dist += pow(pt1[i] - pt2[i], 2)
    return dist

# which cluster the data point belongs to
def which_cluster(features, means):
    dist_min = 100000
    for i in range(len(means)):
        dist = eucli_dist(features, means[i])
        if dist <= dist_min:
            dist_min = dist
            cluster = i
    return cluster

# get the corresponding clusters of entire data
def get_clusters(data, means):
    cluster_val = []
    for i in range(len(data)):
        val = which_cluster(data[i], means)
        cluster_val.append(val)
    return cluster_val

# get unique elements in a list
def unique(list1): 
    unique_list = []  
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

# to update the new cluster means
def update_means(data, cluster_val):
    values = unique(cluster_val)
    values.sort()
    means = []
    for i in range(len(values)):
        count = 0
        lst = []
        for j in range(len(data)):
            if cluster_val[j] == values[i]:
                if len(lst) == 0:
                    for k in range(len(data[j])):
                        lst.append(data[j][k])
                    count = count + 1
                    continue
                for k in range(len(lst)):
                    lst[k] += data[j][k]
                count = count + 1
        for k in range(len(lst)):
            lst[k] /= count
        if len(lst) == 0:
            continue
        means.append(lst)
    return means

# get lists with indices of data in different clusters
def sort_clusters(values):
    c_val = unique(values)
    c_val.sort()
    clusters = []
    for i in range(len(c_val)):
        lst = []
        for j in range(len(values)):
            if values[j] == c_val[i]:
                lst.append(j)
        clusters.append(lst)
    return clusters

# to get intersection of 2 sets
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

# to compute Jacquard Distance
def Jacquard_Dist(lst1, lst2):
    lst3 = intersection(lst1, lst2)
    J_ind = (len(lst3) / (len(lst1) + len(lst2) - len(lst3))) * 100
    J_dist = 100 - J_ind
    return J_dist


data, values = ReadData('data4_19.csv')

# Uncomment the below line to input number of clusters

num_clu = int(input('Enter number of clusters : '))


cluster_means = InitializeMeans(data, num_clu) #initialize random means

# Uncomment the below line to input number of iterations

num_iter = int(input('Enter number of iterations : '))


for i in range(num_iter): 
    cluster_value = get_clusters(data, cluster_means) # get clusters
    cluster_means = update_means(data, cluster_value) # update cluster means
cluster_value = get_clusters(data, cluster_means)

print("") # print final cluster means
for i in range(len(cluster_means)):
	print("Cluster ", i + 1, " mean : ", cluster_means[i])
	print("")

classes = unique(values) # get given cluster data
for i in range(len(classes)):
    for j in range(len(values)):
        if values[j] == classes[i]:
            values[j] = i;

formed_clusters = sort_clusters(cluster_value) # trained clusters
original_clusters = sort_clusters(values) # original clusters

jacq_dist = [] 
for i in range(len(formed_clusters)): # computer Jacquard Distance
    j_dist = []
    for j in range(len(original_clusters)):
        dist = Jacquard_Dist(formed_clusters[i], original_clusters[j])
        j_dist.append(dist)
    jacq_dist.append(j_dist)

print("Jacquard distance between : ") # print Jacquard Distance
print("")
for i in range(len(jacq_dist)):
    print("Cluster ", i + 1, " original clusters : ", jacq_dist[i])
    print("")
    j = jacq_dist[i].index(min(jacq_dist[i]))
    print("Cluster ", i + 1, " ---->  Original Cluster ", j + 1, " (Jacquard distance = ", jacq_dist[i][j], " %)")
    print("")




