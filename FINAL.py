# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 13:59:05 2025

@author: ayesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,precision_score,f1_score
import random
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

##LINEAR REGRESSION
# Load the data
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(housing.feature_names)  # Confirm feature names
print(housing_df.columns)
print(housing.target_names)
# Initiate the variables
X = housing.data[:, 0].reshape(-1, 1)
Y = housing.target.reshape(-1, 1)

# Plot the initial graph
plt.scatter(X, Y, alpha=0.5)
plt.xlabel("Income")
plt.ylabel("House Price")
plt.title("Income vs Houseprice")
plt.show()

# Descriptive statistics
stats=housing_df["MedInc"].describe()
stats2=pd.Series(housing.target).describe()
print(stats)
print(stats2)
# Normalise the independent variable
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
def gradient_descent(X_train,Y_train,iterations=30,lr=0.1):
    n=len(X_train)
    w_curr=0
    b_curr=0
    w=[]
    b=[]
    cost_func=[]
    for iteration in range(iterations):
        y_predicted=X_train*w_curr+b_curr
        dw=-(2/n)*np.sum(X_train*(Y_train-y_predicted))
        db=-(2/n)*np.sum(Y_train-y_predicted)
        w_curr=w_curr-(lr*dw)
        b_curr=b_curr-(lr*db)
        cost=(1/n)*np.sum((Y_train-y_predicted)**2)
        print(f"Iteration{iteration},w={w_curr},b={b_curr},cost={cost}")
        w.append(w_curr)
        b.append(b_curr)
        cost_func.append(cost)
    return(w,b,cost_func)
w,b,cost_func=gradient_descent(X_train,Y_train)


plt.plot(cost_func)

plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Cost Function Convergence for BGD")
plt.show()

w_final_bgd=w[-1]
b_final_bgd=b[-1]
y_predbgd = (X_test*w_final_bgd) + b_final_bgd
mse_bgd = mean_squared_error(Y_test, y_predbgd)
r2score_bgd=r2_score(Y_test,y_predbgd)
print(f"Mean Squared Error for batch gradient descent: {mse_bgd}")
print(f"R2 Score for batch gradient descent:{r2score_bgd}")
plt.scatter(X_test, Y_test, alpha=0.5, label="Actual")
plt.plot(X_test, y_predbgd, color='red', label="Prediction")
plt.xlabel("Income (Normalised)")
plt.ylabel("House Price")
plt.title("Prediction vs Actual")
plt.legend()
plt.show()

X_new=scaler_X.transform([[8]])
Y_pred_new = (X_new * w_final_bgd) + b_final_bgd  
print(f"Predicted house price for median income of 8: {Y_pred_new*100000}")

def stochastic_gradient_descent(X_train, Y_train, iterations=500, lr=0.01):
    n = len(X_train)
    w_curr_sgd = 0
    b_curr_sgd = 0
    w_sgd = []
    b_sgd = []
    cost_func_sgd = []
    
    for iteration in range(iterations):
        # Shuffle the data
        indices = np.arange(n)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        total_cost =0
        
        for i in range(n):
            # Select a single data point
            X_i = X_train_shuffled[i]
            Y_i = Y_train_shuffled[i]
            
            # Predict the value
            y_predicted = X_i * w_curr_sgd + b_curr_sgd
            
            # Compute gradients
            dw = -2 * X_i * (Y_i - y_predicted)
            db = -2 * (Y_i - y_predicted)
            
            # Update weights and biases
            w_curr_sgd = w_curr_sgd - (lr * dw)
            b_curr_sgd = b_curr_sgd - (lr * db)
            
            # Compute cost for the current data point
            cost2 = (1/n)*np.sum((Y_i - y_predicted) ** 2)
            # Append values for tracking
            w_sgd.append(w_curr_sgd)
            b_sgd.append(b_curr_sgd)
            
        cost_func_sgd.append(cost2) 
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, w={w_curr_sgd}, b={b_curr_sgd}, cost={cost_func_sgd[-1]}")
    
    return w_sgd, b_sgd, cost_func_sgd

# Run stochastic gradient descent
w_sgd, b_sgd, cost_func_sgd = stochastic_gradient_descent(X_train, Y_train)

# Final weights and biases
w_final = w_sgd[-1]
b_final = b_sgd[-1]

# Make predictions on the test set
y_predsgd = (X_test * w_final) + b_final

# Calculate MSE for Stochastic gradient descent
mse_sgd = mean_squared_error(Y_test, y_predsgd)
r2score_sgd=r2_score(Y_test,y_predsgd)
print(f"Mean Squared Error for Stochastic Gradient Descent: {mse_sgd}")
print(f"R2 Score for Stochastic Gradient Descent:{r2score_sgd}")

# Plot the cost function convergence
plt.plot(cost_func_sgd)
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Cost Function Convergence (SGD)")
plt.show()

# Plot the predictions vs actual values
plt.scatter(X_test, Y_test, alpha=0.5, label="Actual")
plt.plot(X_test, y_predsgd, color='red', label="Prediction")
plt.xlabel("Income (Normalised)")
plt.ylabel("House Price")
plt.title("Prediction vs Actual (SGD)")
plt.legend()
plt.show()

# Predict house price for a new data point
X_new = scaler_X.transform(np.array([[8]]))  
Y_pred_new = (X_new * w_final) + b_final
print(f"Predicted house price for median income of 8: {Y_pred_new[0][0] * 100000}")


##SVM
Seed=48
np.random.seed(Seed)
random.seed(Seed)


PATH = r"C:\AI\HAR\UCI HAR Dataset\UCI HAR Dataset"  

#Outline the data files
features_path = PATH + r"\features.txt"
activity_labels_path = PATH + r"\activity_labels.txt"
X_train_path = PATH + r"\train\X_train.txt"
y_train_path = PATH + r"\train\y_train.txt"
X_test_path = PATH + r"\test\X_test.txt"
y_test_path = PATH + r"\test\y_test.txt"

#Loading feature names & handling duplicates
features_df = pd.read_csv(features_path, delim_whitespace=True, header=None, names=["idx", "feature"])
features_df["feature"] = features_df["feature"] + "_" + features_df["idx"].astype(str)
feature_names = features_df["feature"].tolist()

#Load activity labels
activity_labels_df = pd.read_csv(activity_labels_path, delim_whitespace=True, header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

#Load train/test datasets
X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None, names=["Activity"])
X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None, names=["Activity"])

#Map activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)


def to_binary_label(activity):
    return 1 if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"] else 0

y_train["Binary"] = y_train["Activity"].apply(to_binary_label).astype(int)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label).astype(int)

#Initial data exploration & visualisations
class_distribution=y_train["Binary"].value_counts()
print(f"The class distribution in the training set is:{class_distribution}")
labels = ["Inactive", "Active"]
plt.figure(figsize=(4,4))
plt.pie(class_distribution,labels=labels,autopct='%1.1f%%',startangle=90, colors=["royalblue", "mediumpurple"],shadow=True)
plt.title("Train Set Class Distribution")
plt.show()
class_distribution2=y_test["Binary"].value_counts() 
print(f"The class distribution in the test set is: {class_distribution2}")
plt.figure(figsize=(4,4))
plt.pie(class_distribution2,labels=labels,autopct='%1.1f%%',startangle=90, colors=["darkorange", "crimson"],shadow=True)
plt.title("Test Set Class Distribution")
plt.show()


#Linear kernel
start_time = time.time()
svml=SVC(kernel="linear",random_state=Seed)
svml.fit(X_train,y_train["Binary"])
end_time = time.time()
predsvml=svml.predict(X_test)
predsvml
accuracy_l=accuracy_score(y_test["Binary"],predsvml)
precision_l=precision_score(y_test["Binary"],predsvml)
f1_score_l=f1_score(y_test["Binary"],predsvml)
print(f"Accuracy for the linear kernel is: {accuracy_l:.4f}")
print(f"Training time for Linear SVM: {end_time - start_time:.4f} seconds")
print(f"Precision for linear kernel: {precision_l}")
print(f"F1 Score for linear kernel:{f1_score_l}")


#Polynomial kernel
start_time=time.time()
svmp=SVC(kernel="poly",random_state=Seed)
svmp.fit(X_train,y_train["Binary"])
end_time=time.time()
predsvmp=svmp.predict(X_test)
accuracyp=accuracy_score(y_test["Binary"],predsvmp)
precisionp=precision_score(y_test["Binary"],predsvmp)
f1_score_p=f1_score(y_test["Binary"],predsvmp)
print(f"Accuracy for the polynomial kernel is: {accuracyp:.4f}")
print(f"Training time for Polynomial SVM: {end_time - start_time:.4f} seconds")
print(f"Precision for the polynomial kernel:{precisionp}")
print(f"F1 Score for the polynomial kernel:{f1_score_p}")

#Radial basis function kernel
start_time=time.time()
svmrbf=SVC(kernel="rbf",random_state=Seed)
svmrbf.fit(X_train,y_train["Binary"])
end_time=time.time()
predsvmrbf=svmrbf.predict(X_test)
accuracyrbf=accuracy_score(y_test["Binary"],predsvmrbf)
precisionrbf=precision_score(y_test["Binary"],predsvmrbf)
f1_score_rbf=f1_score(y_test["Binary"],predsvmrbf)
print(f"Accuracy for the Radial basis function kernel is: {accuracyrbf:.4f}")
print(f"Training time for RBF SVM: {end_time - start_time:.4f} seconds")
print(f"Precision for the radial basis function:{precisionrbf}")
print(f"F1 Score for RBF kernel:{f1_score_rbf}")



#Accuracies of the kernels
accuracy_results = {
    "Kernel": ["Linear", "Polynomial", "RBF"],
    "Accuracy": [accuracy_l, accuracyp, accuracyrbf]
}

#Convert accuracies to DataFrame
accuracy_df = pd.DataFrame(accuracy_results)
print("Accuracy Results Before Cross-Validation:")
print(accuracy_df.to_string(index=False))
#Gridsearch (all features)
param_grid={
        "C":[0.1,1,10],
        "kernel":["linear","poly","rbf"],
        "gamma":["scale","auto"] }

svm=SVC(random_state=Seed)
start_time = time.time()
grid_search=GridSearchCV(svm,param_grid,cv=3,scoring="accuracy")
grid_search.fit(X_train,y_train["Binary"])
end_time = time.time()
training_time_grid_search = end_time - start_time
print(f"Training time for Grid Search: {training_time_grid_search:.4f} seconds")
print("Best Parameters:",grid_search.best_params_)
best_svm=grid_search.best_estimator_
best_svm.fit(X_train, y_train["Binary"])
preds=best_svm.predict(X_test)
#Accuracy after gridsearch (all features)
accuracy2=accuracy_score(y_test["Binary"], preds)
print(f"Optimised SVM Accuracy: {accuracy2}")
print("Best cross-validation accuracy:", grid_search.best_score_)
#Evaluation metrics
conf_mat=confusion_matrix(y_test["Binary"],preds)
print("Confusion Matrix:", conf_mat)

precision2 = precision_score(y_test["Binary"], preds)
f1 = f1_score(y_test["Binary"], preds)
print(f"Precision: {precision2:.4f}")
print(f"F1 Score: {f1:.4f}")



#Streamlined gridsearch (50 features)
pipeline = Pipeline([("scaler", StandardScaler()),
("pca", PCA(n_components=50,random_state=Seed)),
("svc", SVC(random_state=Seed))])
param_grid2=[{

        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear"]},
{
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["poly"],
        "svc__degree": [2, 3],  
        "svc__gamma": [0.01,0.1,1]},
{
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["rbf"],
        "svc__gamma": [0.01,0.1,1]}]

start_time = time.time()
grid_search2=GridSearchCV(pipeline,param_grid=param_grid2,cv=5,scoring="accuracy")
grid_search2.fit(X_train, y_train["Binary"])
end_time = time.time()
training_time_grid_search_pca = end_time - start_time
print(f"Training time for Grid Search with PCA: {training_time_grid_search_pca:.4f} seconds")

print("Best parameters:", grid_search2.best_params_)
best_svm2=grid_search2.best_estimator_
best_svm2.fit(X_train,y_train["Binary"])
preds2=best_svm2.predict(X_test)
#Accuracy after gridsearch (50 features)
accuracy_gridsearch_pca=accuracy_score(y_test["Binary"],preds2)
print(f"PCA SVM accuracy: {accuracy_gridsearch_pca}")
print("Best cross-validation accuracy:", grid_search2.best_score_)

#Evaluation metrics (pca)
conf_mat2=confusion_matrix(y_test["Binary"],preds2)
print("Confusion Matrix:", conf_mat2)
precision_pca=precision_score(y_test["Binary"], preds2)
f1_pca=f1_score(y_test["Binary"], preds2)
print(f"Precision: {precision_pca:.4f}")
print(f"F1 Score: {f1_pca:.4f}")
