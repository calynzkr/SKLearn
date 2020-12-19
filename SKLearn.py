#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


  
def Kmeans_choose (input_X, K):
    # determine k use k means
    distortions = []
    
    for k in range(1,K):
        kmeanModel = KMeans(n_clusters=k).fit(input_X)
        distortions.append(sum(np.min(cdist(input_X, kmeanModel.cluster_centers_, 'euclidean'), 
                                      axis=1)) / input_X.shape[0])
    
    # plot the elbow method 
    plt.plot(range(1,K), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Using The Elbow Method to Show the Optimal k')
    return plt.show()

if __name__ == "__main__":
    #Question 1 
    boston = datasets.load_boston()
    #fit the linear model
    lm = LinearRegression()
    lm.fit(boston.data, boston.target)
    sve = pd.DataFrame(data=lm.coef_, index=boston.feature_names, columns=['coefficient'])

    # when coefficient gets bigger, the impact on house value gets bigger
    sve = sve.reindex(sve['coefficient'].abs().sort_values(ascending=True).index)
    print(sve)
    
    #Question 2
    iris = datasets.load_iris(return_X_y=True)
    result_2 = Kmeans_choose(iris[0], 10)
    
    wine = datasets.load_wine(return_X_y=True)
    result_3 = Kmeans_choose(wine[0], 10)
    


# In[ ]:




