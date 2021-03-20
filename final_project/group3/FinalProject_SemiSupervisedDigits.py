# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:25:16 2021

@author: prewi
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# Importing the data
    # train_data = number data fro model training
    # test_data = number data to label with trained model
    # sample_data = sample df of labeled test_data dim and format.
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_data = pd.read_csv("sample_submission.csv")


#########################################################                                                       
# Sources
#########################################################
# source: "Data driven model and scientific computation, Methods for 
    # complex systems and big data" by J. Nathan Kutz 
# source: "Python data science handbook, essential tools for working with
    # data" by Jake VanderPlas
# source: 'https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34'        
    # from class reading material.
# source: 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV'                             
# source: 'https://scikit-learn.org/stable/modules/svm.html'
# source: 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'
# source: 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'
##########################################################

## Goal of this approach is to get high classification accuracy with 
    # operational complexity being of secondary importance using a 
    # semi-supervised learning model.  
        # starting with unsupervised learning to lower the dimesionality and
            # remove sparsity.  In addition this allows the SVC to operate in
            # a preprocessed eigenspace of the images. 

# training and testing steps: 
    # 1, create model type and test on small data batch
    # 2, identify range of parameter values via CV
    # 3, interatively test model to see parameter handling, and small
        # data batch accuracy and variations 
    # 4, review small batch results
    # 5, scale to larger data batch 
    # 6, interate to establish models final parameters from trials and find 
        # accuracy/variaions in larger data batch on the order of magnitude 
        # equivalent to the dataset needing labeled
    # 7, apply parameters to final model for use in labeling new images
    # 8, complete multiple runs and confirm variation between outputs matches  
        # expected variations that were established from training trials
    
    
# models to use, PCA and SVC. Utilizing the single value decomp aspect of PCA  
    # to find the eigen vectors of the digit images. Enables the info to 
    # be preserved and the working rank greatly reduced. 
    # Then passing to SVC for classification from the decomposed matrices.    
    
    
# create usable matrices
X = train_data.drop('label', axis=1)
y = train_data[['label']]

# PCA does not standardize/scale  
    # need to standardize values due to range and resultant sensitivities 
scaler = StandardScaler()
scaler.fit_transform(X)




# split into train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, 
                                                    random_state=1234)



# searching for usable number of principle components
pca_fit = PCA().fit(X_train)
plt.plot((np.cumsum(pca_fit.explained_variance_ratio_)))
plt.xlim(-30,830)
plt.xlabel('Number of Primary Components')
plt.ylabel('Explained Variance')
plt.title('PCA Component Investigation')
plt.grid(True)




# creating a model pipeline using principal component analyis and then SVC
    # using 'rbf' because of the general shape of the shapes of many digits
    # using a n_componenets = 81 to give still lets approximately 85% of the ,
    # but also reduce the working dimensions of each  image by an approximate 
    # order of magnitude. This is done by finding the eigen-vectors and 
    # corresponding values, then keeping the number of values which preserve 
    # the proprtion of information from the images.  Once the transformation
    # is complete, the pipeline passes data to SVC for classification
    
pca = PCA(n_components=81)
svc = SVC(kernel='rbf')
model = Pipeline([("PCA", pca),("SVC", svc)])



# establish parameters for search using 1/observations*var default for gamma 
    # searching for optimized C. started with [0.1,100], narrowed interatively
    
params = dict(SVC__C = [1, 5, 15, 25],
              )
# searching for best parameters for the model using 3-fold cross validation
search = GridSearchCV(model, params, cv=3)
search.fit(X_train, y_train.values.ravel())
print(search.best_params_)
print('\n')

#using the best parameters in the model
model = search.best_estimator_
y_best = model.predict(X_test)

# determining the accuracy of the SVC model
accuracy_single_SVM = accuracy_score(y_test, y_best)

print(f"Model: Single Trial Accuracy = {accuracy_single_SVM}")
print('\n')
# confusion matrix of model colorized by count quantity 
    # using range 11 to ensure no errors and anything assigned outside 
conf_mat0 = confusion_matrix(y_test , y_best)
sns.heatmap(conf_mat0.T, square=True, annot=True, fmt='d',
            xticklabels = np.arange(0,11), yticklabels = np.arange(0,11))
plt.title('Param Search Confusion Matrix, 10% Training')
plt.xlabel("True Number Label")
plt.ylabel('Predicted Number Label')





# ##################################################################
# ### Final training and testing with 70% training data
# ##################################################################

## iteratively test model to establish parameter handling, acccuracy,
    # and prediction variations. Larger data batch.

# initializing vectors for loop
yBest = []
accuracySVM =[]
best_est = []

# [] initializing my readiness to wait...   :(

# generating loop to test model over 5 seed values such that statistical
    # analysis can be performed on the resulting C values to find best 
    # parameters which optimize model performance 

    
# creating new random instances
X = train_data.drop('label', axis=1)
y = train_data[['label']]
scaler = StandardScaler()
scaler.fit_transform(X)
   

for i in range(117,122):
    # calling many forms of random seeds in subsetting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, 
                                                        random_state = i)
    model = Pipeline([("PCA", pca),("SVC", svc)])
    search = GridSearchCV(model, params)
    search.fit(X_train, y_train.values.ravel())
    model = search                # switching back to 'model' for readability 
    model.best_estimator_         # use best estimator for seed
    # storing each iteration's used best parameter
    best_est.append(model.best_estimator_._final_estimator.C)
    y_trialB = model.predict(X_test)
    accuracy_trial = accuracy_score(y_test, y_trialB)
    yBest.append(y_trialB)
    accuracySVM.append(accuracy_trial)
    
# showing the trial accuracies
# print('\n')
print(f"Model: All 5 Trials Accuracies, 70% Train: {accuracySVM}")
print('\n')
# evaluating accuracies in training trials 
    # mean
avg_accuracy = statistics.mean(accuracySVM)
    # interquartile range in training trials
accuracy_IQR = (statistics.quantiles(data=accuracySVM, n=4)[2] - 
              statistics.quantiles(data=accuracySVM, n=4)[0])

# # establishing C estimator in model to be used from SVC trials
C_trained =statistics.mean(best_est)
print(f"Trained C estimator = {C_trained}")
                                  

print(f"Model Average Accuracy = {statistics.mean(accuracySVM)}")     
print(f"Model Average Accuracy IQR = {accuracy_IQR}")
print('\n')
                     
# confusion matrix of model colorized by count quantity from final trial
conf_mat2 = confusion_matrix(y_test , y_trialB)
sns.heatmap(conf_mat2.T, square=True, annot=True, fmt='d',
            xticklabels = np.arange(0,10), yticklabels = np.arange(0,10))
plt.title('Confusion Matrix for 70% Training Data')
plt.xlabel("True Number Label")
plt.ylabel('Predicted Number Label')


print("Training Complete!!!")
print("Time to use it on 'test_data' for labeling!")
print('\n')




####################################################
# The trained model
####################################################

pca_final = PCA(n_components=81)
svc_final = SVC(kernel='rbf', C = C_trained)
model_final = Pipeline([("PCA", pca_final),("SVC", svc_final)]).fit(X_train, 
                                                        y_train.values.ravel())

#####################################################
#####################################################


####################################################
# Final test of the model on data ~ same saize as unknown with new random input
####################################################

# creating new random instances
X = train_data.drop('label', axis=1)
y = train_data[['label']]
scaler = StandardScaler()
scaler.fit_transform(X)


# testing model with distilled knowledge  

y_final = []
final_accuracy = []

for i in range(817,822):
    # calling many forms of random seeds in subsetting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, 
                                                        random_state = i)
    y_tester = model_final.predict(X_test)
    final_accuracy_trial = accuracy_score(y_test, y_tester)
    y_final.append(y_tester)
    final_accuracy.append(final_accuracy_trial)   
 
# acurracy statistics    
avg_accuracy_final = statistics.mean(final_accuracy)
    # interquartile range in training trials
accuracy_final_IQR = (statistics.quantiles(data=final_accuracy, n=4)[2] - 
              statistics.quantiles(data=final_accuracy, n=4)[0])

print(f"Trained model average accuracy = {avg_accuracy_final}")
print(f"Trained model accuracy IQR = {accuracy_final_IQR}")


# testing model with distilled knowledge  
y_final_Sample = model_final.predict(X_test)
# tesing the accuracy
final_accuracy_Sample = accuracy_score(y_test, y_final_Sample)

# confusion matrix of model colorized by count quantity from final trial
conf_mat3 = confusion_matrix(y_test , y_final_Sample)
sns.heatmap(conf_mat3.T, square=True, annot=True, fmt='d',
            xticklabels = np.arange(0,10), yticklabels = np.arange(0,10))
plt.title('Confusion Matrix for Trained Model')
plt.xlabel("True Number Label")
plt.ylabel('Predicted Number Label')

# trained model classification report
print(classification_report(y_test, y_final_Sample))


####################################################
# Applying the model to final set
####################################################

# using 'test_data' as predictor in trained model to classify digits
Classified_Digits = model_final.predict(test_data)
# predicting twice to ensure no variations due to float values
Classified_Digits2 = model_final.predict(test_data)



# checking quality of data and that it has no errors
bool_check = [Classified_Digits == Classified_Digits2]
print(f"There are missed values: '{any(np.isnan(Classified_Digits))}'")
print(f"Predictions are unchanging between instances: '{all(bool_check[0])}'") 
print(f"Minimum of Classified Digit Values = {min(Classified_Digits)}") 
print(f"Maximum of Classified Digit Values = {max(Classified_Digits)}") 
print('\n')
checker = all(bool_check[0])


# adding assertions to ensure data integrity
assert len(Classified_Digits) == 28000
assert min(Classified_Digits) == 0
assert max(Classified_Digits) == 9
assert checker == True


# formating data to match expected style
Classified_Digits = sample_data
Classified_Digits["Label"] = model_final.predict(test_data)

# final quality assertion
assert Classified_Digits.shape == (28000,2)
print('Quality checks passed!')
print('\n')


# writing data as CSV for turn in. 
Classified_Digits.to_csv('Final_ClassifiedDigits.csv')

print("Data stored in WORKING DIRECTORY as Final_ClassifiedDigits.csv")
print("Classification task complete!!!")



##################################################
### Verify outputs match expectations 
##################################################



# read written csv files
    # 4 different instances of labeling 'sample_data'. 
    # each instance evaluated from knowlege distilled from different training 
        # seed trials
            # CD1 = seeds(1XX)
            # CD2 = seeds(2XX)
            # CD3 = seeds(3XX)
            # CD4 = seeds(4XX)
# CD1 = pd.read_csv("Final_ClassifiedDigits.csv")
# CD2 = pd.read_csv("Final_ClassifiedDigits2.csv")
# CD3 = pd.read_csv("Final_ClassifiedDigits3.csv")
# CD4 = pd.read_csv("Final_ClassifiedDigits4.csv")

# # call only labeled digits
# C1 = CD1['Label']
# C2 = CD2['Label']
# C3 = CD3['Label']
# C4 = CD4['Label']


# # find index values where the vectors do not equal eachother
# variation1 = pd.DataFrame(np.where(C1 != C2)).T  
# variation2 = pd.DataFrame(np.where(C1 != C3)).T  
# variation3 = pd.DataFrame(np.where(C3 != C2)).T
# variation4 = pd.DataFrame(np.where(C1 != C4)).T
# variation5 = pd.DataFrame(np.where(C2 != C4)).T
# variation6 = pd.DataFrame(np.where(C3 != C4)).T

# # illustrating magnitudes of output differences
# print(f"Number of variaitons between 1:2 = {len(variation1)}") 
# print(f"Number of variaitons between 1:3 = {len(variation2)}") 
# print(f"Number of variations between 3:2 = {len(variation3)}") 
# print(f"Number of variations between 1:4 = {len(variation4)}") 
# print(f"Number of variations between 2:4 = {len(variation5)}") 
# print(f"Number of variations between 3:4 = {len(variation6)}") 
# print('\n')
# print(f"Variation between 1:2 = {len(variation1)/len(sample_data)}") 
# print(f"Variation between 1:3 = {len(variation2)/len(sample_data)}") 
# print(f"Variation between 3:2 = {len(variation3)/len(sample_data)}") 
# print(f"Variation between 1:4 = {len(variation4)/len(sample_data)}")
# print(f"Variation between 2:4 = {len(variation5)/len(sample_data)}") 
# print(f"Variation between 3:4 = {len(variation6)/len(sample_data)}")  
# print('\n')

# # matrix of each realtive accuracy
# Racc_mat = [len(variation1)/len(sample_data), len(variation2)/len(sample_data),
#             len(variation3)/len(sample_data), len(variation4)/len(sample_data),
#             len(variation5)/len(sample_data), len(variation6)/len(sample_data)]

# # finding model relative accuracy for comparison to accuracy 
# model_Raccuracy = 1 - statistics.mean(Racc_mat)
# model_Raccuracy_IQR = (statistics.quantiles(data=Racc_mat, n=4)[2] - 
#                        statistics.quantiles(data=Racc_mat, n=4)[0])

# print(f"Relative Output Accuracy = {model_Raccuracy}")
# print(f"Accuracy Expectation = {avg_accuracy}")
# print(f"IQR of Relative Output Accuracy = {model_Raccuracy_IQR}")
# print(f"IQR Expectation = {accuracy_IQR}")
# print('\n')

# print("Digit Classification Result Comparison Fisnished!!!")
