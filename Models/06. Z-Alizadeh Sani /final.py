#%%
## Import libraries
import numpy as np
import pandas as pd
from sklearn import feature_selection

# Setting print options
pd.options.display.float_format = "{:,.2f}".format
np.printoptions(precision=2)
#%%
## Read CSV
df = pd.read_csv('Z-Alizadeh sani dataset.csv', delimiter=",")
df.head(10)
#%%
### Generate descriptive statistics
## Descriptive statistics include those that summarize the central
# tendency, dispersion and shape of a dataset's distribution, excluding ``NaN`` values.
df.describe()
# df.nunique()
#%%
## Function body of plotPerColumnDistribution
from helpers import categorical_to_ordinal, plotPerColumnDistribution
import inspect
print(inspect.getsource(plotPerColumnDistribution))


# %%
plotPerColumnDistribution(df, 10, 2)
# %%
## Display  NULL NaN values (if there are any)
df.isna()
df.columns[df.isna().any()].tolist() # Shows columns with NaN values
# %%
# Since we are training random forest classifier, we need to do two changes:
# 1. Convert binary class columns to ordinal: Yes - 1, No - 0
# 2. Drop columns that have multiple categorical data (few exceptions)

Sex = {'Male': 1,'Fmale': 0}
df.Sex = [Sex[item] for item in df.Sex]
df.Obesity = categorical_to_ordinal(df.Obesity)
df.DLP =  categorical_to_ordinal(df.DLP)
df.Dyspnea =  categorical_to_ordinal(df.Dyspnea)
df.Atypical = categorical_to_ordinal(df.Atypical)
df.Nonanginal = categorical_to_ordinal(df.Nonanginal)

#%%
## Drop columns with Categorical data
df=df.drop(['CRF','CVA','Airway disease','Thyroid Disease','CHF','Weak Peripheral Pulse','Lung rales','Systolic Murmur','Diastolic Murmur','Exertional CP','LowTH Ang','Q Wave','St Elevation','LVH','Poor R Progression','VHD'], axis=1)
df.head(5)
#%%%
## Split dataset into X and Y 
X, y = df.iloc[:, :-1], df.iloc[:, -1]
print("Input data with columns", X.shape)
print("Predictions shape: ", y.shape)
#%%
# Dividing dataset into train and test set
from sklearn.model_selection import train_test_split

# 0.25 signifies the size of test dataset: 25%
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=1122)
print("Train data shape: X:", X_train.shape, ", Y: ", Y_train.shape)
print("Test data shape: X:", X_test.shape, ", Y: ", Y_test.shape)

# %%
from sklearn.tree import DecisionTreeClassifier

model0 = DecisionTreeClassifier(random_state=687)
model0.fit(X_train, Y_train)
prediction0 = model0.predict(X_test)
#%%
### From parsing through results
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("The accuracy score of Decision tree (model0) is: ", round(accuracy_score(Y_test, prediction0)*100, 2))
# %%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model1 = DecisionTreeClassifier(random_state=687)
model1.fit(X_train, Y_train)
prediction1 = model1.predict(X_test)
print("The accuracy score (After preprocessing) of Decision tree (model1) is: ", accuracy_score(Y_test, prediction1))
# %%

from sklearn.neural_network import MLPClassifier

model2 = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(10, 20, 5,2), random_state=1234)
model2.fit(X_train, Y_train)
prediction2 = model2.predict (X_test)
print("The accuracy score (After preprocessing) of MLP is: ", round(accuracy_score(Y_test, prediction2)*100, 2))

# %%
from sklearn.naive_bayes import GaussianNB

model3 = GaussianNB()
model3.fit(X_train, Y_train)
prediction3 = model3.predict(X_test)
print("The accuracy score (After preprocessing) of Gaussian Naive Bayes is: ", accuracy_score(Y_test, prediction3))
print(classification_report(Y_test, prediction3))
# %%
from sklearn.ensemble import VotingClassifier
model4 = VotingClassifier(estimators=[('dtc1', model0), ('dtc2', model1), ('mlp', model2), ('gnb', model3)],
                        voting='soft',
                        weights=[1, 1, 2, 1])



model4.fit(X_train, Y_train)
# loop through algorithms and append the score into the list model.fit(X_train, y_train)
prediction4 =model4.predict(X_test)
#score = model.score(X_test, y_test)
print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction4)))
print (classification_report(Y_test,prediction4))
# %%
## HARD Voting classifier
model5 = VotingClassifier(estimators=[('dtc1', model0), ('dtc2', model1), ('mlp', model2), ('gnb', model3)],
                        voting='hard',
                        weights=[1, 1, 2, 1])



model5.fit(X_train, Y_train)
# loop through algorithms and append the score into the list model.fit(X_train, y_train)
prediction5 =model5.predict(X_test)
#score = model.score(X_test, y_test)
print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction5)))
print (classification_report(Y_test,prediction5))
# %%
#%%
### ENSEMBLE -2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rfc = RandomForestClassifier(n_estimators=200, random_state=5678)
model_feature_selection = SelectFromModel(rfc)
model_feature_selection.fit(X,y)
model_feature_selection.get_support()
selected_features = X.columns[model_feature_selection.get_support()]
print("Number of selected features: ", len(selected_features))
print("Selected features are: ", list(selected_features))
# %%
## Modifying our test data and splitting
X = X[selected_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=1122)
print("Train data shape: X:", X_train.shape, ", Y: ", Y_train.shape)
print("Test data shape: X:", X_test.shape, ", Y: ", Y_test.shape)
#%%
model2_0 = DecisionTreeClassifier(random_state=687)
model2_0.fit(X_train, Y_train)
prediction2_0 = model2_0.predict(X_test)
### From parsing through results
print("The accuracy score of Decision tree (model2_0) is: ", accuracy_score(Y_test, prediction2_0))
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model2_1 = DecisionTreeClassifier(random_state=687)
model2_1.fit(X_train, Y_train)
prediction1 = model2_1.predict(X_test)
print("The accuracy score (After preprocessing) of Decision tree (model2_1) is: ", accuracy_score(Y_test, prediction1))
# %%
# MLP Classifer
model2_2 = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(5,5,2), random_state=1234)
model2_2.fit(X_train, Y_train)
prediction2 = model2_2.predict (X_test)
print("The accuracy score (After preprocessing) of MLP is: ", accuracy_score(Y_test, prediction2))

# %%
## Gaussian Naive Bayes Classifer
model2_3 = GaussianNB()
model2_3.fit(X_train, Y_train)
prediction3 = model2_3.predict(X_test)
print("The accuracy score (After preprocessing) of Gaussian Naive Bayes is: ", accuracy_score(Y_test, prediction3))
print(classification_report(Y_test, prediction3))
# %%
## Ensemble classifer
model2_4 = VotingClassifier(estimators=[('dtc1', model2_0), ('dtc2', model2_1), ('mlp', model2_2), ('gnb', model2_3)],
                        voting='soft',
                        weights=[1, 1, 2, 2.5])



model2_4.fit(X_train, Y_train)
# loop through algorithms and append the score into the list model2_.fit(X_train, y_train)
prediction4 =model2_4.predict(X_test)
#score = model2_.score(X_test, y_test)
print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction4)))
print (classification_report(Y_test,prediction4))
# %%
## HARD Voting classifier
model2_5 = VotingClassifier(estimators=[('dtc1', model2_0), ('dtc2', model2_1), ('mlp', model2_2), ('gnb', model2_3)],
                        voting='hard',
                        weights=[1, 1, 2, 2.5])



model2_5.fit(X_train, Y_train)
# loop through algorithms and append the score into the list model2_.fit(X_train, y_train)
prediction5 =model2_5.predict(X_test)
#score = model2_.score(X_test, y_test)
print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction5)))
print (classification_report(Y_test,prediction5))
# %%
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state=0, max_iter=10000)
clf_1 = classifier1.fit(X_train, Y_train)
y_pred1 = clf_1.predict(X_test)
print('Accuracy of Logistic Regression is {}'.format(accuracy_score(Y_test,y_pred1 )*100))
#start_time = time.time()
print(classification_report(Y_test,y_pred1))
# %%
# from sklearn.cluster import KMeans
# model2_6 = KMeans(n_clusters=2, random_state=0).fit(X_train)
# abc=model2_6.predict(X_test)
# print('Accuracy of KMeans is {}'.format(accuracy_score(Y_test,abc)*100))
# start_time = time.time()
# print(classification_report(Y_test,y_pred1))

# %%
