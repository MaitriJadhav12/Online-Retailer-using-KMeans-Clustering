#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[3]:


ds=pd.read_csv('OnlineRetail.csv',encoding='ISO-8859-1')


# In[4]:


ds.head()


# In[5]:


ds.tail()


# In[6]:


print(ds.info()) # data cleaning
print(ds.shape)
print(ds.isnull().sum())
ds=ds.dropna()
print(ds.info())
print(ds.shape)


# In[7]:


ds['CustomerID']=ds['CustomerID'].astype(str)   # data processing 
ds['Amount']=ds['Quantity']*ds['UnitPrice']
rfm_ds_m=ds.groupby('CustomerID')['Amount'].sum()  # r=recency-no.of days since last purchase
rfm_ds_m.reset_index()                             #f=frequency-no. of transactions
rfm_ds_m.columns=['CustomerID','Amount']           #m=monetary-total no. of transactions
print(rfm_ds_m)


# In[8]:


rfm_ds_f=ds.groupby('CustomerID')['InvoiceNo'].count()
rfm_ds_f=rfm_ds_f.reset_index()
rfm_ds_f.columns=['CustomerID','Frequency']
print(rfm_ds_f)


# In[14]:


ds['InvoiceDate'] = pd.to_datetime(ds['InvoiceDate'],format='%d-%m-%Y %H:%M')
max_date = max(ds['InvoiceDate'])
ds['Diff'] = max_date - ds['InvoiceDate']
rfm_ds_p = ds.groupby('CustomerID')['Diff'].min()
rfm_ds_p = rfm_ds_p.reset_index()
rfm_ds_p.columns = ['CustomerID','Diff']
rfm_ds_p['Diff'] = rfm_ds_p['Diff'].dt.days
print(rfm_ds_p)


# In[15]:


rfm_ds_final=pd.merge(rfm_ds_m,rfm_ds_f,on='CustomerID',how='inner')
rfm_ds_final=pd.merge(rfm_ds_final,rfm_ds_p,on='CustomerID',how='inner')
rfm_ds_final.columns=['CustomerID','Amount','Frequency','Recency']
print(rfm_ds_final.head())


# In[18]:


Q1=rfm_ds_final.Amount.quantile(0.05)
Q3=rfm_ds_final.Amount.quantile(0.95)
IQR=Q3-Q1
rfm_ds_final=rfm_ds_final[(rfm_ds_final.Amount >=Q1 -1.5*IQR)&(rfm_ds_final.Amount <=Q3+1.5*IQR)]

Q1=rfm_ds_final.Recency.quantile(0.05)
Q3=rfm_ds_final.Recency.quantile(0.95)
IQR=Q3-Q1
rfm_ds_final=rfm_ds_final[(rfm_ds_final.Recency >=Q1 -1.5*IQR)&(rfm_ds_final.Recency  <=Q3+1.5*IQR)]

Q1=rfm_ds_final.Frequency.quantile(0.05)
Q3=rfm_ds_final.Frequency.quantile(0.95)
IQR=Q3-Q1
rfm_ds_final=rfm_ds_final[(rfm_ds_final.Frequency >=Q1 -1.5*IQR)&(rfm_ds_final.Frequency <=Q3+1.5*IQR)]
print(rfm_ds_final.shape)


# In[19]:


#scaling
x=rfm_ds_final[['Amount','Frequency','Recency']]
scaler=MinMaxScaler()
rfm_ds_scaled=scaler.fit_transform(x)


# In[20]:


rfm_ds_scaled=pd.DataFrame(rfm_ds_scaled)
rfm_ds_scaled.columns=['Amount','Frequency','Recency']
rfm_ds_scaled.head()


# In[21]:


#model creation
kmeans=KMeans(n_clusters=3,max_iter=50)
kmeans.fit(rfm_ds_scaled)
lbs=kmeans.labels_
print(kmeans.labels_)


# In[23]:


#wss
wss=[]
range_n_clusters=[2,3,4,5,6,7,8]
for num_clusters in range_n_clusters:
    kmeans=KMeans(n_clusters=num_clusters,max_iter=50)
    kmeans.fit(rfm_ds_scaled)
    wss.append(kmeans.inertia_)
plt.plot(wss)


# In[24]:


#silhouette score
range_n_clusters=[2,3,4,5,6,7,8]
for num_clusters in range_n_clusters:
    kmeans=KMeans(n_clusters=num_clusters,max_iter=50)
    kmeans.fit(rfm_ds_scaled)
    cluster_labels=kmeans.labels_
    silhouette_avg=silhouette_score(rfm_ds_scaled,cluster_labels)
    print("For n_clusters={0},the silhouette score is {1}".format(num_clusters,silhouette_avg))


# In[25]:


rfm_ds_final['Cluster_Id']=lbs
rfm_ds_final.head()


# In[27]:


sns.boxplot(x='Cluster_Id',y='Amount',data=rfm_ds_final)


# In[28]:


sns.boxplot(x='Cluster_Id',y='Frequency',data=rfm_ds_final)


# In[29]:


sns.boxplot(x='Cluster_Id',y='Recency',data=rfm_ds_final)


# In[ ]:




