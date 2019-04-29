
# coding: utf-8

# # Basic Algorithmic Learning

# ## Linear Regression Models

# In[1]:


#This is a machine learning algorithm you can use to quantify and make predictions based on relationships between 
#two numerical values

#This model has a number of assumptions:
#Data is continuous and numerical
#Data is not missing any values and does not have any outliers
#Theres a linear relationship between predictors and predicants
# All predictors are independent of each other, no correlation with each other
#Residuals(prediction errors) are normally distributed


# In[1]:


#Lets import our libraries

import numpy as np
import seaborn as sb
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from collections import Counter


# In[3]:


#Lets set our plotting parameters

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')


# In[139]:


#Lets import the dataset
link='/Users/afatade/Desktop/Data Science with Python/Ex_Files_Python_Data_Science_EssT/Exercise Files/Ch08/08_01/enrollment_forecast.csv'
enroll=pd.read_csv(link)


# In[140]:


len(enroll)


# In[5]:


enroll.head()


# In[6]:


#Lets generate a scatterplot matrix to see what variables have a linear relationship
sb.pairplot(enroll)


# In[7]:


#The linearity between unem and roll could be stronger, but we will call this good enough to use for predictions
#each of our variables is also continuously numeric!

#We will be making use of the unem and hgrad columns. But we want to make sure there is NO CORRELATION
#between our PREDICTORS.
enroll.corr()


# In[8]:


#We can see that correlation between h grad and unem is very low. Great!
#Lets split our data into features and targets

enroll_data=enroll.iloc[:, [2,3]].values
target=enroll.iloc[:, 1].values
enroll_data_names=['unem', 'hgrad']

#It is also good to scale our data before feeding it into a classifier
X=scale(enroll_data)
y=target


# In[9]:


X


# ### Check for missing values

# In[10]:


#Lets check our data if it has any missing values
data=pd.DataFrame(X)
data.isnull().sum()


# In[11]:


#We have no missing values


# In[12]:


#Lets instantiate a linear regression model. We should always set normalize to true. This tells the model to 
#normalize our variables before regression

LinReg=LinearRegression(normalize=True)
LinReg.fit(X,y)


# In[13]:


#Lets print out a score for our model
print(LinReg.score(X,y))


# In[14]:


#We have 84% accuracy. A larger dataset would yield more insights


# In[15]:


LinReg.predict([[ 0.34682081, -2.42562243]])


# In[16]:


enroll.head()


# In[49]:


val=enroll_data[1:5]


# In[50]:


val


# In[55]:


#What if you want to use new data that has to be scaled? You can use the standard scaler library to remember the
#scaling of your training data.

#Fit in the unscaled data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(enroll_data)


# In[57]:


val=scaler.transform(val)


# In[59]:


LinReg.predict(val)


# In[62]:


val=[[8.1,9552]]
val=scaler.transform(val)


# In[63]:


LinReg.predict(val)


# ## Logistic Regression

# In[64]:


#Logistic regression is a simple machine learning algorithm you can use to predict numeric categorical variables.
#This differs from linear regression, as in this case, you are predicting categories for ordinal variables, while in 
#linear regression, you are predicting numerical variables for numeric continous variables

#It takes assumptions such that data is free from missing values
#There are at least 50 observations for reliable results
#All predictors are independent of each other
#The predictant variable is binary(there are only two) or a categorical variable with ordered values

#Lets import our libraries

import numpy as np
import seaborn as sb
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

from collections import Counter


# In[65]:


#Lets set our plotting parameters

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')


# ## Logistic Regression on mtcars

# In[2]:


link='/Users/afatade/Desktop/Data Science with Python/Ex_Files_Python_Data_Science_EssT/Exercise Files/Ch08/08_02/mtcars.csv'
cars=pd.read_csv(link)
cars.head()


# In[3]:


cars.columns


# In[4]:


cars.columns=['Car Type', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs',
       'am', 'gear', 'carb']


# In[5]:


cars.head()


# In[75]:


corrs=cars.corr(method='spearman')


# In[76]:


sb.heatmap(corrs)


# In[74]:


sb.pairplot(cars, hue='am', palette='hls')


# In[77]:


#Since we want to predict a category, such as am, a logistic regression model should work.
#But what variables shall we use?

#We have to pick two variables which must be categorical variables with ordered data.
#We see that mpg, wt, qsec, disp and hp all seem to be continuous numeric variables. Lets filter them 
#out and generate another heatmap.
cars.columns


# In[79]:


cars_subset=cars.iloc[:,[2,5,8,9,10,11]]
cars_corr=cars_subset.corr(method='spearman')
sb.heatmap(cars_corr)


# In[6]:


cars_subset=cars.iloc[:,[2,5,8,9,10,11]]
cars_subset.head()


# In[15]:


cars_subset.groupby(['am'])


# In[17]:


cars_groups.mean()


# In[ ]:


#It is worth noting, that carb, gear, drat and cyl have some form of correlation with am. 
#Lets say we decide to pick two. Which ones will we pick?
#They're all categorical variables, but we want to make use of predicants that are independent of each other
#and have more of a spread of categories.


# In[82]:


cars['carb'].value_counts()


# In[83]:


cars['cyl'].value_counts()


# In[84]:


cars['vs'].value_counts()


# In[85]:


cars['drat'].value_counts()


# In[86]:


cars['gear'].value_counts()


# In[87]:


#We see that carb and drat have a wider range of categorical data so we will use those two. 
#But are they independent of each other?

cars_subset.head()


# In[88]:


#Lets reduce our data further
cars_data=cars_subset.iloc[:,[1,5]]
cars_data.head()


# In[89]:


cars_data.corr(method='spearman')


# In[90]:


#We see that they have little to no correlation to one another! They're good to go.
#Lets also print out our data on a scatterplot
sb.regplot(x='drat', y='carb', data=cars_data)


# In[91]:


#We can see that the data we are working with are ordinal variables! as they are categorized


# ### Check for No missing values

# In[92]:


cars_data.isnull().sum()


# In[93]:


#We see we have no missing values in our dataset.


# ### Check if your target variable is binary or ordinal

# In[95]:


#Our target variable is am. Lets see if it is binary or ordinal.
#We can use a countplot for this.

sb.countplot(x='am', data=cars, palette='hls')


# In[96]:


#Great! We can see that we have a binary target variable.
#Now lets check if our data meets the 50 count record requirement.

cars.info()


# In[97]:


#We see we only have 32 entries. This could be a potential problem so in the future we want to make sure
#that the data we make use of has at least 50 entries


# ### Deploying and Evaluating your model

# In[102]:


#Now that we have confirmed that these variables will work, lets scale our data and split it into training
#and testing data

X=scale(cars_data.values)
y=cars.iloc[:,9].values

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.33)


# In[105]:


y_test


# In[106]:


#Lets instantiate and fit the model
LogReg=LogisticRegression()
LogReg.fit(X_train, y_train)


# In[108]:


#Lets print out our score
LogReg.score(X,y)


# In[110]:


#We have an accuracy of 87.5%
#Lets use the classsification report to base our model on precision and recall
yPred=LogReg.predict(X)
print(metrics.classification_report(y,yPred))


# In[111]:


#We have 90% precision and 88% completeness. This is a good model.


# ## Naive Bayes Classifier

# In[112]:


#A naive bayes classifier is a machine learning algorithm you can use to predict the likelihood that an 
#event will occur given evidence thats present in your data

#There are three types of Naive Bayes classifier- Multinomial, Bernoulli and Gaussian.

#Multinomial-Good for when your features,(categorical or continuous) describe discrete frequency counts, e.g word count
#Bernoulli= Good for making predictions from binary values
#Gaussian- Good for making predictions on normally distributed data.


# In[113]:


#The model has assumptions such that are predictors are independent of one another, 
#and your data has an a priori assumption: this is an assumption that the past condition holds true. When we make prediction
#from historical values, we will get incorrect results if present circumstances change

#All regression models maintain an a priori assumption.

#Lets import our libraries.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

import urllib

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Using Naive Bayes to predict spam

# In[115]:


#Our dataset can be found online so lets use the urllib library to read it in.
url='https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
raw_data=urllib.request.urlopen(url)


# In[117]:


#Lets convert it to some data using numpy. We will read it in as a txt file, but we remember to use a comma delimiter
dataset=np.loadtxt(raw_data, delimiter=',')
dataset[0]


# In[120]:


#Taking a closer look at our dataset we can see it seems to be standardized, so we dont have to scale it.
#Furthermore, we want to work with variables taht desscribe discrete frequency counts e.g word counts.
#The first 48 variables in this dataset relate to word count, so lets use that data as our input
#Our target variable seems to be binary; either 1 or 0 so we set the last column to our target.

X=dataset[:,0:48]
y=dataset[:,-1] #quick way to access the last column in our datasset


# In[123]:


len(X[0])


# In[128]:


#Lets split our data into testing and training data
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.33, random_state=7)

#Our dataset set compromises of continous variables that describe frequency word count, so multinomial is our 
#best option. However, we can also try Bernoulli with binning to convert frequency counts to binary values 
#and Guassian too.


# In[129]:


BernNB= BernoulliNB(binarize=True) #This will bin our data into binary data
BernNB.fit(X_train, y_train)

yPred=BernNB.predict(X_test)
print(accuracy_score(y_test, yPred))


# In[130]:


MultiNB=MultinomialNB()
MultiNB.fit(X_train, y_train)

yPred=MultiNB.predict(X_test)
print(accuracy_score(y_test, yPred))


# In[131]:


Gauss= GaussianNB()
Gauss.fit(X_train, y_train)
yPred=Gauss.predict(X_test)
print(accuracy_score(y_test, yPred))


# In[132]:


#We see that the mulinomial NB has the highest level of accuracy, but the other classifiers perform well.


# In[133]:


#What if we modify our bin width in the Bernoulli Naive Bayes classifier?

BernNB= BernoulliNB(binarize=0.1) #This will bin our data into binary data
BernNB.fit(X_train, y_train)

yPred=BernNB.predict(X_test)
print(accuracy_score(y_test, yPred))


# In[ ]:


#We now get an accuracy of 88.15. It would be best to make use of the NaiveBayes for larger datasets and
#it would be wise to evaluate all three models.

