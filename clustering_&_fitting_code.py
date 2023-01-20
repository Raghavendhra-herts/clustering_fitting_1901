# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:27:22 2023

@author: Raghavendhra Rao Devineni
"""
# student ID: 21072747
# GitHub Repository: https://github.com/Raghavendhra-herts/clustering_fitting_1901


# Clustering and Fitting

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit
from numpy import arange
import scipy.optimize as opt


def prepare_data(datafile):
    '''
    In this function I'm preparing to read the dataset in a .csv format taken 
    from the Worldbank
    and returning the dataset 

    '''
    data = pd.read_csv(datafile)
    # printing the first five rows
    print(data.head())
    # finding the NaN/null values and fill the into 0
    data = data.fillna(0)
    # dropping the un-necessary columns
    data = data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code','Unnamed: 66'])

    '''
    print(data.head())
    print(data.dtypes)
    # print("\n", data.columns)
    # print(data.isna().sum())
    '''
    return data

def transpose_data(data):
    '''
    
    In this function I'm passing the attriburte get_t_data, it will send the data
    to this function for transposing the dataset.
    and return the new dataframe

    '''
    # transpose the data set
    df_t = pd.DataFrame.transpose(data)
    # getting the 0th position values to get-into list
    new_header = df_t.iloc[0].values.tolist()
    # reassign them into columns
    df_t.columns = new_header
    # get the data after from the 1: position
    df_t = df_t.iloc[1:]
    # return the variable
    return df_t

# creating the heap_map for the dataset
def heat_map_corr(data, size = 10):
    '''   
    this function is to create a heatmap for the correlation matrix for each column
    in the dataframe
    '''
    # getting the correlation for the data
    corr = data.corr()
    # setting the figure sixe and cordinates for the heatmap and plotting the graph
    figure, axis = plt.subplots(figsize = (size, size))
    # sending the correlation values to the map
    axis.matshow(corr, cmap = 'coolwarm')
    # setting the ticks to the column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.legend()
    plt.show()

# plt.show()


def show_clustering(selected_data, data):
    # scatter plot for the dataset before clustering
    pd.plotting.scatter_matrix(selected_data, figsize=(10, 10), s = 5, alpha = 0.8)
    # showing the plot to see the data
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
        
    plt.title("plotting the clustering to the GDP PER CAPITA dataset")
    plt.xlabel('Year 2020', size = '11')
    plt.ylabel('Year 2021', size = '11')
    plt.legend()
    plt.show()
    
    
    df_centers = pd.DataFrame(cls_centers[:,1:3], columns = ['2016', '2021'])
    print("\n","centroids for both the years: ","\n",df_centers)
    
    # data  = data.groupby('Country Name').sum()



# curve-fitting

def show_curve_fit(data):
   
    def exponential(x, l, m ,n):
        '''
        this is the refression model , it works by adding the squared terms to 
        the exponential function
        where x is an dependent variable
        l & m are the coeffiecient x & y
        n is the constant
        '''
        exp = l * x + m * x**2 + n
        return exp
    
    # check and drop the Nan values in the dataset
    data.dropna(inplace=True)
    # preparing the dataframe by selecting particular columns
    cf_data = data[['Country Name','2012','2016', '2020', '2021']]
    # getting numpy representation of the dataframe
    cf_data = cf_data.values
    # choosing the input & output values
    x , y = cf_data[:, 4], cf_data[:,-1]
    # applying the curve_fit to above inputs
    params, _ = curve_fit(exponential, x ,y)
    # print(params)
    # get the summarize of the parameters value from the above fit call
    val1, val2, val3 = params
    
    # printing the optinal parameters value
    expr_y = '%.5f * x + %.5f * x**2 + %.5f' % (val1, val2, val3)
    # print("\n", expr_y)
    
    # plotting the scatter plot by passing the input  & output values( x&y)
    plt.scatter(x, y, label = 'data')
    # identifying the maximum and minimum values from the given inputs
    x_val = arange(min(x), max(x), 1)
    # calculating the output from the range (min and max)
    y_val = exponential(x_val, val1, val2, val3)
    print()
    # for the above mapping function , create a line plot
    plt.plot(x_val, y_val , "--", color='r', label = 'fit')
    plt.title("plot of the polynomial line fit to GDP PER CAPITA dataset")
    plt.xlabel("Years (2012, 2016, 2020)")
    plt.ylabel("Year (2021)")
    plt.legend()
    # showing the plot result
    plt.show()



if __name__ == "__main__":
    # read the dataset into csv format
    # I have chosen Gdp per capita dataset from world bank
    datafile = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv'
    
    # calling the function to prepare the data
    data = prepare_data(datafile)
    print(data,"\n \n \n")
    
    # calling the function to transpose the dataset 
    data_t = transpose_data(data)
    print(data_t,"\n \n \n")
    
    # printing the statistics variable for the data
    print("\n", data.describe())
    # print("\n", data)
    
    # selecting the particular columns from the dataframe using iloc
    selected_data = data.iloc[:,52:63]
    
    # printing the data
    print(data, "\n \n")
    
    # applying correlation to the selected data from dataframe
    correlation = selected_data.corr()
    # print(correlation, '\n')

    # calling the heapmap function 
    heat_map_corr(selected_data)
    
    # calling the clustering function
    show_clustering(selected_data, data)
    
    # calling the curve_fit function
    show_curve_fit(data)