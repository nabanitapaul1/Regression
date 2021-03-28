# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:36:38 2021

@author: Nabanita Paul
"""

# Linear regression (mainly focus on the problem of Multi-Collinearity)

import pandas as pd
import seaborn as sns
#1. Salary dataset

salary_data = pd.read_csv("Salary_Data.csv")
salary_data
salary_data.shape
salary_data.columns


X = salary_data[["YearsExperience","Age"]]
y= salary_data["Salary"]
sns.pairplot(salary_data)
salary_data.corr()

salary_data.iloc[:,:2].corr()


## Linear Regression
import statsmodels.api as sm
X = sm.add_constant(X)
salary_lm = sm.OLS(y,X).fit()
salary_lm.summary()

# Droping the "Age"

X =X.drop(["Age"],axis=1)
X.columns
salary_lm = sm.OLS(y,X).fit()
salary_lm.summary()

#2. Advertisement Dataset

adv_data=pd.read_csv("Advertising.csv")
adv_data.head()
adv_data.columns
X = adv_data[["TV","radio","newspaper"]]
y= adv_data["sales"]
sns.pairplot(adv_data[["TV","radio","newspaper","sales"]])
adv_data.corr()

adv_data[["TV","radio","newspaper","sales"]].corr()


## Linear Regression
import statsmodels.api as sm
X = sm.add_constant(X)
X
adv_lm = sm.OLS(y,X).fit()
adv_lm.summary()

X.drop("newspaper",axis=1, inplace=True)
X
adv_lm = sm.OLS(y,X).fit()
adv_lm.summary()
