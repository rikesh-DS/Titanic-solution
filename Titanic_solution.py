# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:20:23 2021

@author: KumarRik
"""
#Importing all required library
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


data = pd.concat([train,test], sort = False, ignore_index=True)

#Getting info
data.info()

#Data missing section
print(data.isnull().sum())

'''Age, Fare, Cabin and Embarked columns has
missing data. Analyzing the all columns against of the result and according
 will fillup the value or will drop the columns
'''

#Survived column
train.Survived.isnull().sum()
sns.countplot(x = 'Survived', data = train)

#Passenger class
print(data.Pclass.unique())
sns.countplot(x = 'Pclass', data = data)
sns.countplot(x = 'Pclass',hue = 'Survived', data = train)
print(data.groupby('Pclass').count())
data = pd.concat([data,pd.get_dummies(data['Pclass'], prefix = 'pa_class')],axis = 1)


#Sex column
sns.countplot(x = 'Sex', data = train, hue = 'Survived')
train['Survived'][((train['Sex'] == 'male') & (train['Survived'] == 1))].count()
train['Survived'][((train['Sex'] == 'female') & (train['Survived'] == 1))].count()

data = pd.concat([data,pd.get_dummies(data['Sex'], drop_first=True)],axis = 1)

#Name
data['Name'].head()
"Braund, Mr. Owen Harris".split(',')[1].split('.')[0]
data['Salutation'] = data['Name'].apply(lambda x : x.split(',')[1].split('.')[0])
data.groupby(['Salutation','Sex'])['Age'].mean()
grp = data.groupby(['Pclass','Sex'])

data = pd.concat([data,pd.get_dummies(data['Salutation'])],axis = 1)


#Age
sns.countplot(x = 'Age', hue = 'Survived', data = train)
train['Age'][((train['Survived'] == 1) & (train['Sex'] == 'male'))].median()
train['Age'][((train['Survived'] == 1) & (train['Sex'] == 'female'))].median()
data['Age'].fillna(data.groupby(['Salutation','Sex']).Age.transform('median'), inplace = True)
data['Age_range'] = pd.cut(data['Age'],5)
data = pd.concat([data,pd.get_dummies(data['Age_range'], prefix='Age_ra')],axis = 1)


#SibSp -- sibing spouse
data['SibSp'].unique()
train.groupby('SibSp').count()
sns.countplot(x = 'SibSp',data = train,hue = 'Survived')

#Parch Parent children
data['Parch'].unique()
sns.countplot(x = 'Parch', data = train, hue = 'Survived')

#creatinh new column Family
data['Family'] = data['SibSp'] + data['Parch']
data['Family'].unique()
(data['Family']  == 0).sum()
sns.countplot(x = 'Family', data = data, hue = 'Survived')
data = pd.concat([data,pd.get_dummies(data['Family'],prefix= 'family')], axis = 1)

#ticket
data['Ticket'].head()

#Fare
data['Fare'].isnull().sum()
data.groupby(['Pclass','Sex']).Fare.mean()
data['Fare'].fillna(data.groupby(['Pclass','Sex']).Fare.transform('median'), inplace = True)
data['FareRange'] = pd.cut(data['Fare'], 5)
data = pd.concat([data, pd.get_dummies(data['FareRange'], prefix= 'Farerange')], axis = 1)

#Cabin
data['Cabin'].head()
data['Cabin'].fillna('NA', inplace = True)
temp =re.compile("([a-zA-Z]+)")
data['CabinType'] = data['Cabin'].apply(lambda x :temp.match(x).groups()[0] )
data['CabinType'].unique()  
Canbin_dummy = pd.get_dummies(data['CabinType'], prefix= 'Cabin')
Canbin_dummy.head()
data = pd.concat([data, Canbin_dummy], axis = 1) 

###Embarked
data['Embarked'].unique()
data['Embarked'].isnull().sum()
sns.countplot(x = 'Embarked', data = data) 
data['Embarked'].mode()
data['Embarked'].fillna(data.Embarked.mode()[0], inplace = True)
data['Embarked'].isnull().sum()
sns.countplot(x = 'Embarked', data = train, hue = 'Survived')

#convert the Emabarked data into numberical
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
data['Embarked'].unique()
ema_dummies = pd.get_dummies(data['Embarked'],prefix= 'Embarked',drop_first=True)
data = pd.concat([data,ema_dummies], axis = 1)

####### Dropping the columns
data.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin',
           'Embarked','Salutation','Age_range','FareRange','CabinType'], inplace = True, axis = 1)


data.info()

##Data Division
testing_data = data[data['Survived'].isnull()]
training_data = data.dropna()

testing_data = testing_data.drop(['Survived'],axis =1)

x = training_data.drop(['Survived'],axis = 1)
y = training_data['Survived']


#Data Scalling
ss = StandardScaler()
testing_data = ss.fit_transform(testing_data)
training_data = ss.fit_transform(training_data)


#Spliting the data training data for 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 10)

#model selection and training
rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(x_train,y_train)

prediction = rf.predict(x_test)

accuracy =accuracy_score(y_test, prediction)
print(accuracy)












