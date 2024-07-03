#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


df=pd.read_csv('Mall_Customers.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.info()


# In[11]:


sns.distplot(df['Annual Income (k$)'])


# In[12]:


np.max(df['Annual Income (k$)'])


# In[13]:


sns.distplot(df['Spending Score (1-100)'])


# In[16]:


sns.boxplot(df)
plt.xticks(rotation=45)
plt.show()


# In[17]:


sns.pairplot(df)


# In[21]:


df['Gender'].value_counts()


# In[22]:


sns.pairplot(df,hue='Gender')


# In[23]:


df.corr()


# In[26]:


sns.heatmap(df.corr(),annot=True)


# In[27]:


#clustering 


# In[46]:


clustering1=KMeans(n_clusters=3)


# In[47]:


clustering1.fit(df[['Annual Income (k$)']])


# In[48]:


clustering1.labels_


# In[49]:


df['Income Cluster']=clustering1.labels_
df.head()


# In[50]:


df['Income Cluster'].value_counts()


# In[51]:


clustering1.inertia_


# In[52]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[53]:


inertia_scores


# In[54]:


plt.plot(range(1,11),inertia_scores)


# In[56]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[57]:


#bivarient clustering


# In[59]:


df.columns


# In[63]:


clustering2=KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
clustering2.labels_
df['spending and income cluster']=clustering2.labels_


# In[64]:


inertia_score2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_score2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_score2)


# In[70]:


centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']


# In[73]:


sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue='spending and income cluster')
plt.scatter(x=centers['x'],y=centers['y'],marker='*',color='black')


# In[66]:


df.columns


# In[77]:


pd.crosstab(df['spending and income cluster'],df['Gender'],normalize='index')


# In[79]:


df.groupby('spending and income cluster')['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[80]:


#multivarient cluster


# In[81]:


from sklearn.preprocessing import StandardScaler


# In[82]:


scale=StandardScaler()


# In[83]:


df.head()


# In[110]:


df1=pd.get_dummies(df,drop_first=True)


# In[111]:


df1


# In[112]:


df1.columns


# In[113]:


df1=df1[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]


# In[114]:


df1.head()


# In[115]:


df1=scale.fit_transform(df1)


# In[117]:


inertia_score3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(df1)
    inertia_score3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_score3)


# In[118]:


df


# In[120]:


centers = pd.DataFrame(clustering3.cluster_centers_, columns=['x', 'y'])
print("Cluster Centers:\n", centers)


# In[124]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df1_pca = pca.fit_transform(df1)

clustering3 = KMeans(n_clusters=4)
clustering3.fit(df1_pca)
df1_pca_df = pd.DataFrame(df1_pca, columns=['PCA1', 'PCA2'])
df1_pca_df['Cluster'] = clustering3.labels_

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df1_pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.scatter(x=centers['x'],y=centers['y'],marker='*',color='red')
plt.title('Multivariate Clustering using PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()


# In[125]:


#Analysis


# 
