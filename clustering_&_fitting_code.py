# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:27:22 2023

@author: Raghavendhra
"""

# Clustering and Fitting

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet


data = pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv')
print(data.head())
print("\n", data.columns)
data = data.fillna(0)

data = data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code','Unnamed: 66'])

'''
print(data.head())

print(data.head())
print(data.dtypes)
# print("\n", data.columns)
'''
# print(data.isna().sum())

print("\n", data.describe())
print("\n", data)


def heat_map_corr(data, size = 10):
    corr = data.corr()
    figure, axis = plt.subplots(figsize = (size, size))
    axis.matshow(corr, cmap = 'coolwarm')
    # setting the ticks to the column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
    plt.yticks(range(len(corr.columns)), corr.columns)

selected_data = data.iloc[:,52:63]
print(data)
correlation = selected_data.corr()
print(correlation, '\n')
heat_map_corr(selected_data)
plt.show()



# scatter plot
pd.plotting.scatter_matrix(selected_data, figsize=(10, 10), s = 5, alpha = 0.8)
plt.show()

# creating new dataframe with 4 or 5 columns for clustering
data_cls = data[['2012', '2016', '2020', '2021']]
print(data_cls)

# finding and getting the minimum and maximum column by column 
get_max_val = data_cls.max()
get_min_val = data_cls.min()
print("\n", get_max_val, "\t", get_min_val)
data_cls = (data_cls - get_min_val)/(get_max_val - get_min_val)

print("\n", data_cls)


# setting up the clusterers for the number of clusters
no_clusters = 3
kmeans = cluster.KMeans(n_clusters = no_clusters)

# print(kmeans)
# now fir the data using fit() and store the results in the k-means object
# fit is applied on x & y pairs
kmeans = kmeans.fit(data_cls)
# print("\n", kmeans)

# here labels are the number of associated clusters of x & y
labels = kmeans.labels_
# print("\n", labels)

# find & extract the estimated cluster centers
cls_centers = kmeans.cluster_centers_
print("\n","Cluster centers: \n", cls_centers)

'''
# calculating the silhoutte score
sil_score = skmet.silhouette_score(data_cls, labels)
print(sil_score)
'''

plt.figure(figsize=(10.0, 10.0))
colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
# loop over different labels
for i in range(no_clusters):
    plt.plot(data_cls[labels==i]['2016'],
             data_cls[labels==i]['2021'], "o", markersize = 3,color = colours[i])
    plt.legend()

# here I'm showing the cluster centers
for cc in range(no_clusters):
    cx, cy = cls_centers[cc, :2]
    plt.plot(cx, cy, 'dk', markersize = 10)
    
plt.xlabel('2020')
plt.ylabel('2021')
plt.legend()
plt.show()


df_centers = pd.DataFrame(cls_centers[:,1:3], columns = ['2016', '2021'])
print("\n","centroids for both the years: ","\n",df_centers)

# data = data.groupby('Country Name').sum()






'''
# transposing the data
data_tp = pd.DataFrame.transpose(data)
new_header = data_tp.iloc[0].values.tolist()
data_tp.columns = new_header
data_tp = data_tp.iloc[1:]
print(data_tp)

# data1 = data_tp.head(150)



import scipy.optimize as opt

def exponential_function(tm, growth, scale):
    exp = scale * np.exp(growth * (tm-1960))
    return exp

def logistics(tm, growth, scale, time):
    lg = scale / (1.0 + np.exp(-growth * (tm - time)))
    return lg

data = data.groupby().sum()
print(data)
# parameters , covr = opt.curve_fit(exponential_function, data1[])
    '''