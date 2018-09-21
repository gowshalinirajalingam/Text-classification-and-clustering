import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame


df1 = pd.read_csv("G:\\Hackathon\\hackstat- round 1\\New folder\\diamonds.csv")
print(df1)

#pre-processing data

#change types into correct type.
#price is in int type change it into float 
df1.dtypes
df1["price"]=df1["price"].astype("float64")

#handle missing values.
df1.dropna()
#Here there is no missing values.

#Assigning descreate values to categorical data
#cut variable
df1.cut.unique()
df1.clarity.unique()
df1.color.unique()

#cut-> 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'
df1["cut"]=df1["cut"].replace("Fair",1)
df1["cut"]=df1["cut"].replace("Good",2)
df1["cut"]=df1["cut"].replace("Very Good",3)
df1["cut"]=df1["cut"].replace("Premium",4)
df1["cut"]=df1["cut"].replace("Ideal",5)

#clarity-> 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'
df1["clarity"]=df1["clarity"].replace("I1",1)
df1["clarity"]=df1["clarity"].replace("SI2",2)
df1["clarity"]=df1["clarity"].replace("SI1",3)
df1["clarity"]=df1["clarity"].replace("VS2",4)
df1["clarity"]=df1["clarity"].replace("VS1",5)
df1["clarity"]=df1["clarity"].replace("VVS2",6)
df1["clarity"]=df1["clarity"].replace("VVS1",7)
df1["clarity"]=df1["clarity"].replace("IF",8)

#color->'J', 'I', 'H', 'G', 'F', 'E', 'D'
df1["color"]=df1["color"].replace("J",1)
df1["color"]=df1["color"].replace("I",2)
df1["color"]=df1["color"].replace("H",3)
df1["color"]=df1["color"].replace("G",4)
df1["color"]=df1["color"].replace("F",5)
df1["color"]=df1["color"].replace("E",6)
df1["color"]=df1["color"].replace("D",7)


#divide as training set and test set
msk = np.random.rand(len(df1)) < 0.8
traindf=df1[msk]
testdf=df1[~msk]


#descriptive analysis
#-------------------------

traindf.describe(include="all")

#group by
traindf_test=traindf[['color','clarity','price']]
traindf_grp=traindf_test.groupby(['color','clarity'],as_index='False').mean()
traindf_pivot=pd.pivot_table(traindf_grp,index=['color'],values=["price"],columns=['clarity'])
sns.heatmap(traindf_pivot)


#check for for the statistical approach where parametric or non-parametric

#1)check For distribution
#check dependent(price) variable
sns.distplot(traindf["price"],bins=100)

    #price variable is positive skewed.Not normal
  
#check for independent variables
    #categorical variables(cut,color,clarity)
sns.boxplot(x="cut",y="price",data=traindf) 
sns.boxplot(x="color",y="price",data=traindf) 
sns.boxplot(x="clarity",y="price",data=traindf) 

    #Histogram
traindf[['color','clarity','cut']].hist()
plt.show()

    #density plot
traindf[['color','clarity','cut']].plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.show()
    ##TODO##### find conclusion about distribution for categorical variable

#check for numerical(carat,depth,table,x,y,z)
sns.distplot(traindf["carat"],bins=100)

        #carat variable is positive skewed.
sns.distplot(traindf["depth"],bins=100)
        #depth is normally skewed.
        #test for normality
k,p=stats.mstats.normaltest(traindf["depth"])    #k=>z-score    p=>p-value
    #p<0.05 depth is not normally distributed

sns.distplot(traindf["table"],bins=100)
        #table is normally skewed.
        #test for normality
k,p=stats.mstats.normaltest(traindf["table"])
        #p<0.05 depth is not normally distributed


sns.distplot(traindf["x"],bins=100)
sns.distplot(traindf["y"],bins=100)
sns.distplot(traindf["z"],bins=100)
    # here x,y,z are related with table variable.not with y. 
    #so we need to compare it with table variable

    #correlation

correlations=traindf[['x','y','z','table']].corr()    
    # plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,4,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(['x','y','z','table'])
ax.set_yticklabels(['x','y','z','table'])
plt.show()

    #plot scatter plot
    #plot x variable vs table variable
fig = plt.figure()
plt.scatter(x=traindf['x'],y=traindf['table'])
fig.suptitle('Relationship between x and table variable', fontsize=20)
plt.xlabel('x', fontsize=18)
plt.ylabel('table', fontsize=16)

    #plot y variable vs table variable
fig = plt.figure()
plt.scatter(x=traindf['y'],y=traindf['table'])
fig.suptitle('Relationship between y and table variable', fontsize=20)
plt.xlabel('y', fontsize=18)
plt.ylabel('table', fontsize=16)

    #plot z variable vs table variable
fig = plt.figure()
plt.scatter(x=traindf['z'],y=traindf['table'])
fig.suptitle('Relationship between z and table variable', fontsize=20)
plt.xlabel('z', fontsize=18)
plt.ylabel('table', fontsize=16)

    ##TODO######conclude the relationships
 

# As the dependent and independent variables are not normally distributed and there are ordinal variables
# we are using non-parametric approach

#we are converting the dependent variable to categorical variable.
#binning  
binwidth=int ((max(traindf["price"])-(min(traindf["price"])))/3)
bins =range(min(traindf["price"].astype("int"))-1,max(traindf["price"].astype("int"))+1,binwidth)
groupnames=['Low','medium','High']
traindf['price-binned']=pd.cut(traindf["price"],bins,labels=groupnames)
traindf['price-binned']=traindf['price-binned'].astype('str')

testdf['price-binned']=pd.cut(testdf["price"],bins,labels=groupnames)
testdf['price-binned']=testdf['price-binned'].astype('str')
traindf.dtypes
testdf.dtypes

#check for binomial and poisson distribution
#The data set is not time series data.So the distribution is not poisson.
#as there is more than 2 values(low,medium,high) for response variable we need to use multinominal response model.

#Predictive analysis
#-------------------------- 

Xtraindf=traindf[['carat','cut','color','clarity','depth','table','x','y','z']]
Xtestdf=testdf[['carat','cut','color','clarity','depth','table','x','y','z']]

Ytraindf=traindf[['price-binned']]
Ytestdf=testdf[['price-binned']]
Xtraindf.dtypes
Ytraindf.dtypes

lr = LogisticRegression().fit(Xtraindf,Ytraindf)
yhat = lr.predict(Xtestdf)
data.reshape((999,1))
accuracy_score(Ytestdf, yhat)

#93.85% accurate
