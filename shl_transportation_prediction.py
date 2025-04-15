#!/usr/bin/env python
# coding: utf-8

# # Machıne learning ile boşlukları doldur!!!!!

# In[2]:


import pandas as pd
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',700)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import datetime as dt


# In[3]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report


# - Label sütunu için LabelEncoder
# - Data cleaning
# - null olan değerler ya 0 ile doldur yada corr a bak ve ona göre seçip machine learning ile doldur
# - isnull,shape,unique

# - GPS de 4.sütun 0 ise hiçbir uydu yok demektir.
# - En sondaki sütun uydu sayısını gösterir.

# In[3]:


gps_updated=pd.read_csv("GPS_Updated.csv")
cells=pd.read_csv("Cells.csv")
label=pd.read_csv("Label.csv")
location=pd.read_csv("Location.csv")


# ## Wifi

# In[4]:


wifi=pd.read_csv('WiFi_Train.csv').drop(columns=['Unnamed: 0','1','2'],axis=1)


# In[5]:


wifi=wifi.iloc[:,:43]


# In[6]:


# I reset the axis
wifi.columns=range(wifi.shape[1])


# In[81]:


wifi.rename(columns={0:'Epoch time [ms]'},inplace=True)


# In[82]:


wifi.select_dtypes(include='object') # wifi şifresi yada wifi hesap adı olduğundan gereksiz bilgiler bu yüzden drop yaptım


# In[7]:


wifi=wifi.select_dtypes(exclude='object').drop(columns=[40],axis=1)
wifi


# In[8]:


wifi.info()


# In[9]:


miss=wifi.isnull().sum()
miss[miss>0]


# In[144]:


wifi.rename(columns={'Epoch time [ms]':0},inplace=True)


# ### Filling not object columns with regression

# In[10]:


full=wifi[wifi[39].notnull()]
miss=wifi[wifi[39].isnull()]


# In[11]:


full


# In[12]:


full.isnull().sum()


# In[13]:


miss


# In[21]:


miss


# In[14]:


miss.isnull().sum()


# In[15]:


miss.describe()


# In[16]:


full.describe()


# In[163]:


miss[15].value_counts()


# In[17]:


from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[185]:


miss2=miss.copy()
miss2


# In[18]:


def fulling(a):
    
    miss2=miss.copy()
    for col in miss.columns:
        if col!=a:
            miss2[col]=miss2[col].fillna(miss[col].min())
            
    target=miss2.drop(columns=[a],axis=1)
    y=full[[a]]
    x=full.drop(columns=[a],axis=1)
    #x=scaler.fit_transform(x)
    pred=gb.fit(x,y).predict(target)
    return pred


# In[19]:


for a in full.columns[::-1]:
    miss.loc[:,a]=fulling(a)


# In[20]:


miss.isnull().sum()


# In[24]:


wifi[wifi[39].isnull()]=miss


# In[25]:


wifi.isnull().sum()


# In[26]:


wifi[0]=wifi[0].astype(np.int64) 


# In[27]:


wifi.rename(columns={0:'Epoch time [ms]'},inplace=True)


# In[28]:


wifi


# #### If we fill the object columns: Filling [:,6:42:5] 

# In[50]:


k=wifi.iloc[:,6::5]
k


# In[51]:


k.mode()


# In[52]:


k.columns


# In[53]:


k.isnull().sum()


# In[54]:


for col in k.columns:
    wifi[col]=wifi[col].fillna(k[col].mode()[0])


# In[55]:


wifi.iloc[:,6::5].isnull().sum()


# In[62]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for col in k.columns:
    wifi[col]=lb.fit_transform(wifi[col])


# In[63]:


wifi.iloc[:,6::5]


# In[64]:


wifi


# In[76]:


wifi[8].unique()


# In[74]:


wifi.info()


# ## Location (911109 x n_var)

# In[4]:


location


# In[5]:


location['Epoch time (ms)'].dtype


# In[6]:


pd.to_datetime(location['Epoch time (ms)'], unit='ms')


# - Tüm csv leri en son tek bir train dosyasında toplamamız gerekiyor. Ve bu dosyaları birleştirirken "Time" sütununu baz alacağız bu nedenle tüm dosyalardaki time sütunları aynı cinsten olmalıdır.

# In[7]:


location['Epoch time (ms)']=location['Epoch time (ms)'].astype(np.int64) 


# In[8]:


location['Epoch time (ms)'].dtype


# In[ ]:


location=location.drop(columns=['Ignore','Ignore.1'],axis=1)


# In[9]:


location


# In[10]:


location.isnull().sum()


# In[11]:


location.info()


# ## GPS 1322749 x n_var

# In[12]:


gps_updated


# In[13]:


gps_updated.info()


# In[14]:


gps_updated.isnull().sum()


# In[15]:


gps_updated=gps_updated.iloc[:,:66]# 1 milyondan fazla boş veri olan sütunları attım


# In[16]:


gps_updated.Satellite_Count.unique()


# - Uydu sayısı sıfır ise diğer sütunlar doğal olarak boş oluyor.
# - Uydu sayısı sıfırdan farklı olduğu zaman bu sütunlar uydu değerleri ile dolu oluyor.
# - Bu değerleri 0 ile doldurmak en mantıklısı çünkü olmayan uyduların değerleri sıfırdır.

# In[17]:


idx=gps_updated[gps_updated.Satellite_Count!=0]
idx


# In[18]:


# uydu sayısının 0 olmadığı sütunlardaki boşlukları o sütunun median değeri ile doldurdum
for col in idx.columns:
    gps_updated[col] = gps_updated[col].fillna(gps_updated[col].median()) 


# In[19]:


none_satellite=gps_updated[gps_updated.Satellite_Count==0]
none_satellite


# In[20]:


# uydu sayısının 0 olduğu sütunları ise 0 ile doldurdum
for col in none_satellite.columns:
    gps_updated[col] = gps_updated[col].fillna(0) 


# In[21]:


miss=gps_updated.isnull().sum()
miss[miss>0]


# In[22]:


gps_updated.Satellite_Count.unique()


# ## Cells 1324881 x n_var

# In[ ]:


#cells=pd.read_csv("Cells.csv")


# In[23]:


cells.rename(columns={'0':'Time','1':'Ignore','2':'Ignore','3':'Number of entries'},inplace=True)


# In[24]:


cells=cells.drop(columns=['Ignore','Ignore'],axis=1)


# In[ ]:


cells


# In[25]:


cells.info()


# In[26]:


miss=cells.isnull().sum()
miss[miss>0]


# - 1 milyondan fazla boşluğu olan sütunları attım çünkü zaten veri 1.324.881

# - Peki kalan boşluklar nasıl doldurulmalı?

# ### cells dosyasında sadece ilk 13 sütunu alıp kullanırsak:

# In[27]:


cells3=cells.iloc[:,:13]


# In[28]:


cells3.info()


# In[29]:


cells3.isnull().sum()


# In[30]:


cells3['4'].value_counts()


# In[31]:


cells3['4']=cells3['4'].fillna('LTE')# 4.sütunu en çok tekrar eden sayı ile yanı o sütunun mode değeri ile doldurdum


# In[41]:


from numpy import *


# In[36]:


cells3.isnull().sum()


# In[40]:


cells3['4'].dtype


# In[43]:


# object olmayan sütunları ise o sütunun median değerleri ile doldurdum
for col in cells3.columns:
    if (col!='4'): # for not object type
        cells3[col] = cells3[col].fillna(cells3[col].median()) 


# In[45]:


cells3.isnull().sum()


# In[44]:


cells3


# ### cells dosyasından sadece 44 sütunu alıp kullanmak istersek

# In[46]:


cells2=cells.iloc[:,:44]
cells2


# In[47]:


cells2.isnull().sum()


# In[48]:


cells2.info()


# In[49]:


cells2.Time.isnull().sum()


# In[53]:


#Object olmayan sütunlardaki boşlukları o sütunun median değeri ile doldurdum
for col in cells2.select_dtypes(exclude=['O']).columns:# for not object type
     cells2[col] = cells2[col].fillna(cells2[col].median()) 


# In[55]:


#sadece object olan sütunlardaki boş verilere baktım
df_num = cells2.select_dtypes(include=['object'])
df_num.isnull().sum()


# In[56]:


df_num.mode()#tüm object satırların mode değerlerine baktım


# In[57]:


df_num.describe()


# In[58]:


cells2['4'].unique()


# In[59]:


cells2['4']=cells2['4'].fillna('LTE')


# In[ ]:


cells2['14'].value_counts()


# In[ ]:


cells2['14'].unique()


# In[ ]:


cells2['14'].value_counts()


# In[ ]:


cells2['14'].replace({'0':'LTE'},inplace=True)


# In[ ]:


cells2['14']=cells2['14'].fillna('LTE')


# In[ ]:


cells2['22'].unique()


# In[ ]:


cells2['22'].value_counts()# GSM i ve diğer yazı ile yazılanları olduğu sütundaki değerlerin mode u ile boşluklarıda median değiştirsem??


# In[ ]:


#cells2['22'].replace({'GSM':-113,'WCDMA':-113,'LTE':-113},regex=True,inplace=True)


# In[ ]:


#cells2['22']=cells2['22'].fillna(cells2['22'].median())


# In[ ]:


#cells2['22']=cells2['22'].astype('float')


# In[ ]:


cells2['13'].value_counts()


# In[ ]:


cells2['13'].unique()


# In[ ]:


#cells2['13'].replace({'GSM':2.0,'WCDMA':0.0,'LTE':0.0,'0':0.0},regex=True,inplace=True)


# In[ ]:


#cells2['13']=cells2['13'].fillna(cells2['13'].median())


# In[ ]:


#cells2['13']=cells2['13'].astype('float')


# In[ ]:


cells2['23'].value_counts()


# In[ ]:


cells2['24'].value_counts()


# In[ ]:


cells2['31'].value_counts()


# In[ ]:


cells2['32'].value_counts()


# In[ ]:


cells2['33'].value_counts()


# In[ ]:


cells2['34'].value_counts()


# In[ ]:


cells2['42'].value_counts()


# In[ ]:


cells2['4'].dtype


# - Object olan sütunların boşluklarını , her  sütunun kendi mode değeri ile doldurdum.

# In[ ]:


for col in cells2.columns:
     if (cells2[col].dtype == dtype('O')): # for object type
        cells2[col] = cells2[col].fillna(cells2[col].mode()) 


# In[ ]:


#object sütundaki boş veriler doldu
df_num = cells2.select_dtypes(include=['object'])
df_num.isnull().sum()


# In[ ]:


#sadece sayısal sütunlarda boş veri kaldı
cells2.isnull().sum()


# In[ ]:


#boş verilerin hepsi doldu
miss=cells2.isnull().sum()
miss[miss>0]


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# ## Label 980527 x 2

# In[60]:


label


# In[61]:


label=pd.read_csv("Label.csv",header=None)
label.columns=['Epoch time [ms]','Label']


# In[62]:


label


# In[63]:


label.isnull().sum()


# In[64]:


label.info()


# In[65]:


label.Label.unique()


# - Temizlenmiş ve doldurulmuş dosyaları pickle olarak kaydettim ki tekrar temizlememiz gerekmesin

# In[66]:


import pickle
#location.to_pickle('location_pickle.csv')
cells3.to_pickle('cells_pickle_13column.csv')
#label.to_pickle('label_pickle.csv')
gps_updated.to_pickle('gps_pickle.csv')


# In[ ]:




