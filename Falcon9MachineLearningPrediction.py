#Building and testing different classifiers: KNN, SVM, Logistic Regression and Decision Tree to predict the success/failure of landing 

#1 import libraries 
import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])

# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
#from sklearn.metrics import accuracy_score

#2 define the method to plot the confusion matrix.
def plot_confusion_matrix(y,y_predict):
    #this function plots the confusion matrix
    cf=confusion_matrix(y,y_predict)
    disp=ConfusionMatrixDisplay(confusion_matrix=cf)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    '''
    #this function plots the confusion matrix in another way
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 
    '''
 #3 load dataset
from js import fetch
import io
#original dataset
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = await fetch(URL1)
text1 = io.BytesIO((await resp1.arrayBuffer()).to_py())
data = pd.read_csv(text1)
data
#one hot encoded dataset
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = await fetch(URL2)
text2 = io.BytesIO((await resp2.arrayBuffer()).to_py())
X = pd.read_csv(text2)
X
# asign values to Y - create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y
numpy_class=data['Class'].to_numpy()
Y=pd.DataFrame(numpy_class)
Y=Y.rename(columns={0: 'Class'})
Y

#4 Standardize the data in X then reassign it to the variable X using the transform provided below
#transform = preprocessing.StandardScaler()
X=preprocessing.StandardScaler().fit_transform(X)

#5 Train test split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=42)

'''
#6 Develop Logistc Regression model and use GridSearchCV (cv=10) to find best parameters
#6.1 Create logistic regression object
lr = LogisticRegression()

#6.2 Use GridSearchCV
#define parameters grid
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}

#find the best model using Logistic Regression as estimator
best_model_lr=GridSearchCV(estimator=lr,param_grid=parameters,cv=10,,scoring='accuracy', verbose=2)
best_model_lr.fit(X_train,Y_train.values.ravel()) #without .values.ravel() there is an error message
Y_pred_lr=best_model_lr.predict(X_test)

#6.3. print the best score and the best parameters
print('The best score for Logistic regresion is: ', best_model_lr.best_score_) #0.8625
print('The best parameters for Logistic regresion are: ', best_model_lr.best_params_) #{'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}

#6.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_lr) 
#True Postive - 12 (True label is landed, Predicted label is also landed)
#False Postive - 4 (True label is not landed, Predicted label is landed)

#6.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_lr.score(X_test,Y_test)) #0.7777777777777778

#7. Develop Support Vector Machine model and use GridSearchCV object svm_cv with cv = 10
#7.1 Create SVM object
svm=SVC()
#7.2 Use GridSearchCV
#define parameters grid
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}

best_model_svm=GridSearchCV(estimator=svm,param_grid=parameters, cv=10, scoring='accuracy', verbose=2)
best_model_svm.fit(X_train,Y_train.values.ravel())
Y_pred_svm=best_model_svm.predict(X_test)

#7.3. print the best score and the best parameters
print("The best parameters for SVM", best_model_svm.best_params_) #{'C': 1.0, 'gamma': 0.03162277660168379, 'kernel': 'sigmoid'}
print("The best score for SVM :",best_model_svm.best_score_) #0.85

#7.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_svm)
#True Postive - 12 (True label is landed, Predicted label is also landed)
#False Postive - 3 (True label is not landed, Predicted label is landed)

#7.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_svm.score(X_test,Y_test)) #0.8333333333333334

#8 Develop Decission Tree Classifier and use GridSearchCV (cv=10) to find best parameters
#8.1.Create Decission Tree Classifier Object
tree=DecisionTreeClassifier(random_state=42) #needed to set random_state to 42 because results were different each time the code was executed
#8.2. Use GridSearchCV to find the best model
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}
best_model_tree=GridSearchCV(estimator=tree, param_grid=parameters,cv=10)
best_model_tree.fit(X_train,Y_train.values.ravel())
Y_pred_tree=best_model_tree.predict(X_test)

#8.3. print the best score and the best parameters
print("The best parameters for Decission Tree: ", best_model_tree.best_params_) #{'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
print("The best score for Decission Tree:",best_model_tree.best_score_) #  0.8892857142857142

#8.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_tree)
#True Postive - 11 (True label is landed, Predicted label is also landed)
#False Postive - 4 (True label is not landed, Predicted label is landed)

#8.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_tree.score(X_test,Y_test)) #0.8333333334

#9 Develop K Nearest Neighbors model and use GridSearchCV (cv=10) to find best parameters
#9.1.K Nearest Neighbors Object
knn = KNeighborsClassifier()
#9.2. Use GridSearchCV to find the best model
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

best_model_knn=GridSearchCV(estimator=knn, param_grid=parameters, cv=10, verbose=2)
best_model_knn.fit(X_train,Y_train.values.ravel())
Y_pred_knn=best_model_knn.predict(X_test)

#9.3. print the best score and the best parameters
print("The best parameters for KNN: ", best_model_knn.best_params_) #{'algorithm': 'auto', 'n_neighbors': 8, 'p': 1}
print("The best score for KNN:",best_model_knn.best_score_) #0.8785714285714287

#9.4. plot confusion matrix
plot_confusion_matrix(Y_test,Y_pred_knn)
#True Postive - 12 (True label is landed, Predicted label is also landed)
#False Postive - 4 (True label is not landed, Predicted label is landed)

#9.5. calculate the accuracy on the test data using the method score:
print('Accuracy on TEST data is: ', best_model_knn.score(X_test,Y_test)) #0.777777777777778

'''
'''
The difference between best_score_ and score in the context of using GridSearchCV is as follows:

best_score_: This attribute of the GridSearchCV object represents the best mean cross-validated score achieved during the grid search process. It is calculated using the training data and the cross-validation folds specified. Essentially, it tells you how well the model performed on the training data with the best-found hyperparameters.

score: This method is used to evaluate the performance of the trained model on a separate test dataset. It provides the accuracy (or another scoring metric if specified) of the model on the test data, which is not used during the training or hyperparameter tuning process.

In summary, best_score_ gives you an idea of the model's performance during the training phase with cross-validation, while score provides the model's performance on unseen test data. If you have further questions or need more clarification, feel free to ask!
'''

# Conclusion: Support Vector Machine model with parameters {'C': 1.0, 'gamma': 0.03162277660168379, 'kernel': 'sigmoid'} 
# and Decission Tree with {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'splitter': 'random'} 
# have the same result on test data - 0.83333333334, but SVM has better True positive prediction

'''
#Bar plot showing accuracy achieved by machine learning alghoritms
accuracy_list=[0.77777777777777,0.8333333333333334,0.8333333333333334,0.77777777777777]
algorithm_list=['KNN','SVM','DT','Log Regression']
plt.bar(algorithm_list, accuracy_list)
plt.title('Accuracy Achieved by Machine Learning Alghoritms')
plt.ylabel('Accuracy')
plt.xlabel('Machine Learning Alghoritm')
plt.show()
'''

X_train
