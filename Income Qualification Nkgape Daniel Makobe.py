#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[4]:


train.shape


# In[5]:


test.shape


# In[7]:


train.head(15)


# In[8]:


test.head(2)


# In[9]:


# The Output Variable is Target
print("The output variable is: Target")


# In[10]:


# Understanding the type of data we are dealing with.


# In[11]:


train.info()


# In[13]:


# categorial data 
train.select_dtypes('object').head()


# In[14]:


train["idhogar"].value_counts()


# In[16]:


train["dependency"].value_counts()


# In[18]:


train['edjefe'].value_counts()


# In[19]:


train['edjefa'].value_counts()


# In[22]:


print("the ID and idhogar are identifiers hence we drop them as they will not contribute to the model")
print("dependency, edjefe, edjefa are mixed of categorial data and numerical data")
print("Hence we convert the categorial data to numerical data")


# In[25]:


# convert to numeric as c_t_n
def c_t_n(i):
    if i == "yes":
        return(float(1))
    elif i == "no":
        return(float(0))
    else:
        return(float(i))


# In[26]:


train['dependency']=train['dependency'].apply(c_t_n)
train['edjefe']=train['edjefe'].apply(c_t_n)
train['edjefa']=train['edjefa'].apply(c_t_n)


# In[27]:


train.info()


# In[28]:


train.select_dtypes('object').head()


# In[29]:


print("The two categorial data are identifiers or IDs, we drop them as they wont contribute to the model")


# In[30]:


# Now checking if theres is any biases in the data set
print("we do this by using statistical tools") 


# In[34]:


import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[35]:


print("There is always a bias in a dataset")
print("Machine learning is a representation of probability density functions correlated to each other")
print("We can use statistical tools to compare and see how other variables are correlated")
print("We can achieve this by using hypothesis testing")
print("Null Hypothesis: There is a relation between the variables")
print("Alternative Hypothesis : There is no relationship between the variables")


# In[36]:


data_cont_table = pd.crosstab(train["bedrooms"] , train["overcrowding"])
table = data_cont_table
stat, p, dof, expected = chi2_contingency(table)
print("dof=%d" % dof)
print(expected)
probability = 0.99
critical = chi2.ppf(probability, dof)
print("probability=%.3f, critical=%.3f, stat=%.3f" % (probability, critical, stat))
if abs(stat) >= critical:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")
alpha = 1.0 - probability
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")


# In[37]:


data_cont_table = pd.crosstab(train["r4h1"] , train["r4h3"])
table = data_cont_table
stat, p, dof, expected = chi2_contingency(table)
print("dof=%d" % dof)
print(expected)
probability = 0.99
critical = chi2.ppf(probability, dof)
print("probability=%.3f, critical=%.3f, stat=%.3f" % (probability, critical, stat))
if abs(stat) >= critical:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")
alpha = 1.0 - probability
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")


# In[38]:


data_cont_table = pd.crosstab(train["r4h1"] , train["bedrooms"])
table = data_cont_table
stat, p, dof, expected = chi2_contingency(table)
print("dof=%d" % dof)
print(expected)
probability = 0.99
critical = chi2.ppf(probability, dof)
print("probability=%.3f, critical=%.3f, stat=%.3f" % (probability, critical, stat))
if abs(stat) >= critical:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")
alpha = 1.0 - probability
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")


# In[39]:


data_cont_table = pd.crosstab(train["v18q"] , train["v18q1"])
table = data_cont_table
stat, p, dof, expected = chi2_contingency(table)
print("dof=%d" % dof)
print(expected)
probability = 0.99
critical = chi2.ppf(probability, dof)
print("probability=%.3f, critical=%.3f, stat=%.3f" % (probability, critical, stat))
if abs(stat) >= critical:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")
alpha = 1.0 - probability
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print("Reject the Null Hypothesis")
    print("There is a relation between the variables")
else:
    print("Fail to reject The Null Hypothesis")
    print("There is no relationship between the variables")


# In[40]:


print("There is a bias in the data set.")


# In[41]:


# CHECKING IF ALL MEMBERS OF THE HOUSE HOLD HAVE THE SAME POVERTY LEVEL.


# In[43]:


unique_values = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
different_households = unique_values[unique_values != True]
print('There are {} households where members of the house dont have the same poverty level.'.format(len(different_households)))


# In[45]:


# CHECK IF THERE IS A HOUSE WITHOUT A FAMILY HEAD.
family_head = train.groupby('idhogar')['parentesco1'].sum()
no_head = train.loc[train['idhogar'].isin(family_head[family_head == 0].index), :]
print('There are {} households without a family head.'.format(no_head['idhogar'].nunique()))


# In[46]:


# Set poverty level of the members and the head of the house within a family.


# In[47]:


for individuals_household in different_households.index:
    true_target = int(train[(train["idhogar"] == individuals_household) & (train["parentesco1"] == 1.0)]["Target"])
    train.loc[train["idhogar"] == individuals_household, "Target"] = true_target
unique_values = train.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
different_households = unique_values[unique_values != True]
print("There are {} households where the family members do not all have the same target.".format(len(different_households)))


# In[78]:


poverty_level = train[train["v2a1"] != 0]


# In[79]:


poverty_level.shape


# In[80]:


poverty_level=poverty_level.groupby("area2")["v2a1"].apply(np.median)


# In[81]:


poverty_level


# In[82]:


print("there are Null values in the v2a1 column")
print("v2a1 translates - Monthly rent payment")
print("this means that other family own the houses and dont pay rent")
print("we can replace Null values by 0 rent payment")
train['v2a1'].fillna(0,inplace=True)


# In[83]:


# Count how many null values are existing in columns.
train.isna().sum().value_counts()


# In[94]:


Poverty_level = train[train["v2a1"] != 0]


# In[95]:


Poverty_level.shape


# In[96]:


poverty_level=Poverty_level.groupby("area2")["v2a1"].apply(np.median)
print(poverty_level)


# In[97]:


print("area2 - we see the median of the rent of 140000 then people are below this value in urban area are Below POVERTY LEVEL")
print("area2 - we see the median of rula area of povety line at 80000")


# In[98]:


# we can define a function to filter give poverty levels depending on the rend median values
def povertyID(x):
    if x < 80000:
        return("BELOW POVERTY LEVEL")
    elif x > 140000:
        return("ABOVE POVERTY LEVEL")
    else:
        return("BELOW POVERTY LEVEL OF URBAN AREA, BUT ABOVE POVERTY LEVEL OF A RURAL AREA")


# In[100]:


P_L= Poverty_level["v2a1"].apply(povertyID)


# In[101]:


P_L.shape


# In[102]:


pd.crosstab(P_L,Poverty_level["area2"])


# In[103]:


# Remove null value rows of the target variable
train["Target"].isna().sum()


# In[104]:


print("There is no need to remove null values as they dont exist in this target")


# In[113]:


#Visualising the "Target" column
(train["Target"].value_counts()).head().plot(kind="bar",figsize=(10,7), title = "Poverty target levels")


# In[114]:


train.head(2)


# In[115]:


train.drop(["Id","idhogar"],axis=1,inplace=True)


# In[118]:


X=train.drop("Target",axis=1)
y=train.Target


# In[119]:


# Predict the accuracy using random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[127]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)


# In[128]:


clf = RandomForestClassifier()


# In[141]:


from sklearn.metrics import accuracy_score
model = clf.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_predict,y_test)
print(accuracy)


# In[142]:


# we now clean and fit the test data
#train.select_dtypes('object').head()
test["dependency"]=test["dependency"].apply(c_t_n)
test["edjefe"]=test["edjefe"].apply(c_t_n)
test["edjefa"]=test["edjefa"].apply(c_t_n)
test["v2a1"].fillna(0,inplace=True)


# In[143]:


test_data = imp.fit_transform(test)
test_prediction=model.predict(test_data)
print(test_prediction)


# In[144]:


print("END")


# In[ ]:




