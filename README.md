# fastml

Takes in several inputs regarding the ML problem and carries out pre-processing, model fitting and cross validation to provide accuracy score and comparison between various models.

Inputs :
-------
problem_type : 'Regression' or 'Classification'

models  : 
1. [LinearRegression(), KNeighborsRegressor(), RandomForestRegressor()] choose any of these for regression 
2. [LogisticRegression(), RandomForestClassifier(), KNeighborsClassifier(), GaussianNB(), LinearSVC(), SGDClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier()] choose any of these for classification
           
y       : Column name of the dependant variable as a string 

drops   : List of column names to be removed prior to modeling

tresh_unique   : If the number of unique items in a categorical columns exceed this treshold, the column is dropped
          If the number of unique items in a categorical columns exceed this treshold, the column is not dummified
          
tresh_missing   : If the % of missing items in a column exceed this treshold, the column is dropped

train   : Training set (dataframe)

test    : Testing set (dataframe) is the data on which the model need to be applied
          All the columns (except dependant variable) need to be same as that of the training set
          Test dataset is optional.

Outputs : 
--------
data    : A list of various datasets processed and predicted as part of the analysis
1. data[0] --> X_train - dataframe of features of training set 
2. data[1] --> X_val - dataframe of features of validation set 
3. data[2] --> y_train - dataframe of dependant variable of training set
4. data[3] --> y_val - dataframe of dependant variable of validation set
5. data[4] --> X_test - dataframe of the features of the test set on which the model was applied
6. data[5] --> y_pred - dataframe containing predictions based on the test data set using all models

df_eval : A dataframe comparing accuracy scores among various models.


Examples :
---------
For titanic dataset, 

>>> train = pd.read_csv(path+'train.csv')
>>> test = pd.read_csv(path+'test.csv')

>>> models = [LogisticRegression(), RandomForestClassifier(), KNeighborsClassifier()]
>>> drops = ['PassengerId']

>>> data, df_eval = df_run('Classification', models, 'Survived', drops, 20, 40, train, test)

>>> print(df_eval)

          LogisticRegression  RandomForestClassifier  KNeighborsClassifier
R2_train                0.82                    0.98                  0.79
R2_val                  0.79                    0.80                  0.69
Time                   27.88                  212.43                 34.91
'''
