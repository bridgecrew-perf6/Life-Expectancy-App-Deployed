
import pandas as pd
import numpy as np

import os
import joblib

df=pd.read_csv('led.csv')

df.drop(["Population"],axis=1,inplace=True)

df.drop(['Year'],axis=1,inplace=True)

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan, strategy='median')
df['GDP']=imp.fit_transform(df[['GDP']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Lifeexpectancy'] = imp.fit_transform(df[['Lifeexpectancy']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['AdultMortality'] = imp.fit_transform(df[['AdultMortality']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Alcohol'] = imp.fit_transform(df[['Alcohol']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['HepatitisB'] = imp.fit_transform(df[['HepatitisB']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Alcohol'] = imp.fit_transform(df[['Alcohol']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['BMI'] = imp.fit_transform(df[['BMI']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Totalexpenditure'] = imp.fit_transform(df[['Totalexpenditure']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['thinness1-19years'] = imp.fit_transform(df[['thinness1-19years']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['thinness5-9years'] = imp.fit_transform(df[['thinness5-9years']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Incomecompositionofresources'] = imp.fit_transform(df[['Incomecompositionofresources']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Polio'] = imp.fit_transform(df[['Polio']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Diphtheria'] = imp.fit_transform(df[['Diphtheria']])

imp = SimpleImputer(missing_values = np.nan, strategy='median')
df['Schooling'] = imp.fit_transform(df[['Schooling']])


df_new=df.drop(['Lifeexpectancy','Country','Status'],axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
def vif_calculator(x):
    vif=pd.DataFrame()
    vif["Features"]=x.columns
    vif["VIF"] = [vf(x.values, i)for i in range(len(x.columns))]
    return vif

vif_df=vif_calculator(df_new)


features_to_keep=list(vif_df[vif_df['VIF']<20]['Features'])

df_model=df[features_to_keep]

from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
X=np.array(df[features_to_keep])
y=np.array(df['Lifeexpectancy'])
from sklearn.model_selection import cross_val_score
model=lr()
model.fit(X,y)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.1, normalize=True)
lasso.fit(X,y)
scoresL=cross_val_score(lasso,X,y,cv=8)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.1, normalize=True)
ridge.fit(X,y)
scoresR=cross_val_score(ridge,X,y,cv=8)

file_name='finalized_model.sav'
joblib.dump(ridge,file_name)
loaded_model = joblib.load(file_name)

def Loaded_model(temp):
    temp1=[]
    for i in temp:
        temp1.append(float(i))
    ip=np.array(temp)
    ip=np.reshape(ip,(-1,1)).T
    ip=ip.astype(np.float64)
    r=loaded_model.predict(ip)
    return round(r[0],2)

