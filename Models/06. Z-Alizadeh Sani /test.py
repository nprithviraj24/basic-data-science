#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier
#%%
models = []
models.append(LogisticRegression(random_state=412, max_iter=10000)) # 0th index
models.append(DecisionTreeClassifier(random_state=687)) # 1 index
models.append(MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000,
             hidden_layer_sizes=(5,5,2), random_state=1234)) # 2nd index
models.append(GaussianNB()) # 3rd index
models.append(VotingClassifier(estimators=[('lr', models[0]), ('dtc', models[1]), ('mlp', models[2]), ('gnb', models[3])],
                        voting='soft',
                        weights=[4, 1, 3, 2]))
models.append(VotingClassifier(estimators=[('lr', models[0]), ('dtc', models[1]), ('mlp', models[2]), ('gnb', models[3])],
                        voting='hard',
                        weights=[4, 1, 3, 2]))

#%%

def test_all(X_train, X_test, Y_train, Y_test): # train test split

    for m in models:
        m.fit(X_train, Y_train)
        prediction = m.predict(X_test)
        print (f"The accuracy score of {m.__repr__()}  is {round(accuracy_score(Y_test, prediction)*100,2)}")
        # print (classification_report(Y_test,prediction))
        print("--"*15)
#%%
import pandas as pd
import numpy as np
from helpers import categorical_to_ordinal

df1 = pd.read_csv('Z-Alizadeh sani dataset.csv', delimiter=',')
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)

## Drop columns with Categorical data
df1=df1.drop(['CRF','CVA','Airway disease','Thyroid Disease','CHF','Weak Peripheral Pulse','Lung rales','Systolic Murmur','Diastolic Murmur','Exertional CP','LowTH Ang','Q Wave','St Elevation','LVH','Poor R Progression','VHD'], axis=1)
df1.head(5)

#%%
# Categorical to Ordinal
Sex = {'Male': 1,'Fmale': 0}
df1.Sex = [Sex[item] for item in df1.Sex]

df1.Obesity = categorical_to_ordinal(df1.Obesity)
df1.DLP =  categorical_to_ordinal(df1.DLP)
df1.Dyspnea =  categorical_to_ordinal(df1.Dyspnea)
df1.Atypical = categorical_to_ordinal(df1.Atypical)
df1.Nonanginal = categorical_to_ordinal(df1.Nonanginal)

X, y = df1.iloc[:, :-1], df1.iloc[:, -1]
print("Input data with columns", X.shape)
print("Predictions shape: ", y.shape)
#%%
print("WITHOUT PREPROCESSING")
test_all(*train_test_split(X, y, test_size=0.25, random_state=2606))
#%%
print("WITH PREPROCESSING")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # wHAT IS STANDARD SCALER HERE?
X = scaler.fit_transform(X)
test_all(*train_test_split(X,y, test_size=0.25, random_state=2606))

#%%
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
cols =  df1.iloc[:, :-1].columns # Get the column name
sel.fit( df1.iloc[:, :-1],  df1.iloc[:, -1]) # X, y
# sel.get_support()
selected_feat= df1[cols].columns[(sel.get_support())] # select only True columns
len(selected_feat)
print(selected_feat)
columnsData = df1[selected_feat]
#%%
print ("After preprocessing, and FEATURE SELECTION")
test_all(*train_test_split(columnsData,y, test_size=0.25, random_state=2606))
#%%