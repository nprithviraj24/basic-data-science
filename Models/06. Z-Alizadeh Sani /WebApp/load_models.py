#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from sklearn.cluster import KMeans
import pickle
import os

def load_models():
    datainput = pd.read_csv("traindataset.csv")

    y = datainput['Cath']
    del datainput['Cath']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel

    rfc = RandomForestClassifier(n_estimators=200, random_state=5678)
    model_feature_selection = SelectFromModel(rfc)
    model_feature_selection.fit(datainput,y)
    model_feature_selection.get_support()
    selected_features = datainput.columns[model_feature_selection.get_support()]
    print("Number of selected features: ", len(selected_features))
    print("Selected features are: ", list(selected_features))
    ## Modifying our test data and splitting
    datainput = datainput[selected_features]

    df=datainput
    # Split data into Test & Training set where test data is 20% & training data is 70%
    X_train, X_test, Y_train, Y_test  = train_test_split(datainput, y, test_size=0.2)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    model1 = DecisionTreeClassifier(random_state=687)
    model1.fit(X_train, Y_train)
    prediction1 = model1.predict(X_test)
    print("The accuracy score (After preprocessing) of Decision tree (model1) is: ", accuracy_score(Y_test, prediction1))
    print(classification_report(Y_test, prediction1))

    from sklearn.neural_network import MLPClassifier
    model2 = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(10, 20, 5,2), random_state=1234)
    model2.fit(X_train, Y_train)
    prediction2 = model2.predict(X_test)
    print("The accuracy score (After preprocessing) of MLP is: ", round(accuracy_score(Y_test, prediction2)*100, 2))
    print(classification_report(Y_test, prediction2))

    from sklearn.naive_bayes import GaussianNB
    model3 = GaussianNB()
    model3.fit(X_train, Y_train)
    prediction3 = model3.predict(X_test)
    print("The accuracy score (After preprocessing) of Gaussian Naive Bayes is: ", accuracy_score(Y_test, prediction3))
    print(classification_report(Y_test, prediction3))
    from sklearn.linear_model import LogisticRegression
    clf_lr = LogisticRegression(random_state=0)
    clf_lr.fit(X_train, Y_train)
    y_pred1 = clf_lr.predict(X_test)
    print('Accuracy of Logistic Regression is {}'.format(accuracy_score(Y_test,y_pred1 )*100))
    print(classification_report(Y_test,y_pred1))




    from sklearn.ensemble import VotingClassifier
    model4 = VotingClassifier(estimators=[('LR',clf_lr), ('dtc', model1), ('mlp', model2), ('gnb', model3)],
                            voting='soft',
                            weights=[1, 1, 2, 1])



    model4.fit(X_train, Y_train)
    # loop through algorithms and append the score into the list model.fit(X_train, y_train)
    prediction4 =model4.predict(X_test)
    #score = model.score(X_test, y_test)
    print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction4)))
    print (classification_report(Y_test,prediction4))
    ## HARD Voting classifier
    model5 = VotingClassifier(estimators=[('LR',clf_lr), ('dtc', model1), ('mlp', model2), ('gnb', model3)],
                            voting='hard',
                            weights=[1, 1, 2, 1])



    model5.fit(X_train, Y_train)
    # loop through algorithms and append the score into the list model.fit(X_train, y_train)
    prediction5 =model5.predict(X_test)
    #score = model.score(X_test, y_test)
    print ("The accuracy score of ensemble  is {:.2%}".format(accuracy_score(Y_test,prediction5)))
    print (classification_report(Y_test,prediction5))

    return model1, model2, model3, model4, model5, sc

def save_model(model, name):
    models_path = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    this_path = os.path.abspath(os.path.join(models_path, name))
    # save the classifier
    with open(this_path, 'wb') as fid:
        pickle.dump(model, fid)

#%%

model1, model2, model3, model4, model5, sc =  load_models()
save_model(model4, 'ensemble_soft.pkl')
save_model(sc, 'standard_scaler.pkl')
#%%