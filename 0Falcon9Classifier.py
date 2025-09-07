#Standardize the data
#Split into training data and test data
#Find best Hyperparameter for SVM, Classification Trees and Logistic Regression
#Find the method performs best using test data

import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion_matrix(y,y_predict,classifier):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(classifier); 
    ax.xaxis.set_ticklabels(['will not land', 'will land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 



#2.load the data
from js import fetch
import io
#target var Class
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = await fetch(URL1)
text1 = io.BytesIO((await resp1.arrayBuffer()).to_py())
data = pd.read_csv(text1)
# asign values to Y
Y=data['Class']#make sure to keep one brackets, 1D array, otherwise the Logistic Regression will report an error

#one-hot-encoded dataset - X
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = await fetch(URL2)
text2 = io.BytesIO((await resp2.arrayBuffer()).to_py())
X = pd.read_csv(text2)

#3. train test split, then scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Fit scaler ONLY on training data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform train
# Use same scaler to transform test data (don't fit again!)
X_test_scaled = scaler.transform(X_test)  # Only transform test

#4 Develop Logistic Regression model and use GridSearchCV (cv=10) to find best parameters
#4.1 Create logistic regression object
lr = LogisticRegression()

#4.2 Use GridSearchCV
#define parameters grid
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}

#find the best model using Logistic Regression as estimator
best_model_lr=GridSearchCV(estimator=lr,param_grid=parameters,cv=10, scoring='accuracy', verbose=2)
#best_model_lr.fit(X_train_scaled,Y_train.values.ravel()) #without .values.ravel() there is an error message
best_model_lr.fit(X_train_scaled,Y_train) #if Y is 1D array, and not 2D dataframe, there is no error
Y_pred_lr=best_model_lr.predict(X_test_scaled)

#4.3. print the best score and the best parameters
print('The best score for Logistic regresion is: ', best_model_lr.best_score_) #0.8035714285714285
print('The best parameters for Logistic regresion are: ', best_model_lr.best_params_) #{'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}

#4.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_lr, 'Logistic Regression') 
#True Postive - 14 (Predicted to land, actually landed in reality)
#True Negative - 3 (Predicted to not land, did not land in reality)

#4.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_lr.score(X_test_scaled,Y_test)) # 0.9444444444444444

'''


#5. Develop Support Vector Machine model and use GridSearchCV object svm_cv with cv = 10
#5.1 Create SVM object
svm=SVC()
#5.2 Use GridSearchCV
#define parameters grid
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}

best_model_svm=GridSearchCV(estimator=svm,param_grid=parameters, cv=10, scoring='accuracy', verbose=2)
best_model_svm.fit(X_train_scaled,Y_train.values.ravel())
Y_pred_svm=best_model_svm.predict(X_test_scaled)

#5.3. print the best score and the best parameters
print("The best parameters for SVM", best_model_svm.best_params_) #{'C': 0.03162277660168379, 'gamma': 0.001, 'kernel': 'linear'}
print("The best score for SVM :",best_model_svm.best_score_) #0.8178571428571427

#5.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_svm, 'SVM')
#True Postive - 13 (Predicted to land, landed in reality)
#True Negative - 3 (Predicted to not land, did not land in reality)

#5.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_svm.score(X_test_scaled,Y_test)) #0.88888888888


#6 Develop Decission Tree Classifier and use GridSearchCV (cv=10) to find best parameters
#6.1.Create Decission Tree Classifier Object
tree=DecisionTreeClassifier(random_state=42) #needed to set random_state to 42 because results were different each time the code was executed
#6.2. Use GridSearchCV to find the best model
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

best_model_tree=GridSearchCV(estimator=tree, param_grid=parameters,cv=10)
best_model_tree.fit(X_train_scaled,Y_train.values.ravel())
Y_pred_tree=best_model_tree.predict(X_test_scaled)

#6.3. print the best score and the best parameters
print("The best parameters for Decision Tree: ", best_model_tree.best_params_) #{'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
print("The best score for Decision Tree:",best_model_tree.best_score_) #  0.875

#6.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_tree, 'Decision Tree')
#True Postive - 12 (True label is landed, Predicted label is also landed)
#False Postive - 4 (True label is not landed, Predicted label is landed)

#6.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_tree.score(X_test_scaled,Y_test)) #0.8888888888888


#7 Develop K Nearest Neighbors model and use GridSearchCV (cv=10) to find best parameters
#7.1.K Nearest Neighbors Object
knn = KNeighborsClassifier()
#7.2. Use GridSearchCV to find the best model
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

best_model_knn=GridSearchCV(estimator=knn, param_grid=parameters, cv=10, verbose=2)
best_model_knn.fit(X_train_scaled,Y_train.values.ravel())
Y_pred_knn=best_model_knn.predict(X_test_scaled)

#7.3. print the best score and the best parameters
print("The best parameters for KNN: ", best_model_knn.best_params_) #{'algorithm': 'auto', 'n_neighbors': 7, 'p': 1}
print("The best score for KNN:",best_model_knn.best_score_) #0.832142857142857

#7.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_knn, 'KNN')
#True Postive - 14 (True label is landed, Predicted label is also landed)
#False Postive - 3 (True label is not landed, Predicted label is landed)

#7.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_knn.score(X_test_scaled,Y_test)) #0.944444444444
'''

'''
The difference between best_score_ and score:
best_score_: the best mean cross-validated score achieved during the grid search process. It is calculated using the training data and the cross-validation folds specified. Essentially, it tells you how well the model performed on the training data with the best-found hyperparameters.
score: the performance of the trained model on a separate test dataset. It provides the accuracy (or another scoring metric if specified) of the model on the test data, which is not used during the training or hyperparameter tuning process.
In summary, best_score_ gives you an idea of the model's performance during the training phase with cross-validation, while score provides the model's performance on unseen test data.
'''
'''
#Bar plot showing accuracy achieved by machine learning alghoritms
accuracy_list=[0.944444444444, 0.8888888888888, 0.8888888888888,0.944444444444]
algorithm_list=['KNN','SVM','DT','Log Regression']
plt.bar(algorithm_list, accuracy_list)
plt.title('Accuracy Achieved by Machine Learning Alghoritms')
plt.ylabel('Accuracy')
plt.xlabel('Machine Learning Alghoritm')
plt.show()
'''

# Conclusion: KNN and Log Regression return exactly the same results in terms of accuracy and true predictions

