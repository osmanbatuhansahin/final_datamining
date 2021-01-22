# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler


# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# Bagging
from sklearn.neighbors import KNeighborsClassifier

# Naive bayes
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Library imports
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
# LOADING DATASETS

# LUAD dataset
luad_data = pd.read_csv("TCGA-LUAD_clinical.csv")
# LUSC dataset
lusc_data = pd.read_csv("TCGA-LUSC_clinical.csv")
# LUAD High Risk
luad_high_risk_data = pd.read_csv(
    "LUAD_high_risk_patients.csv")
# LUSC High Risk
lusc_high_risk_data = pd.read_csv(
    "LUSC_high_risk_patients.csv")
# LUAD Low Risk
luad_low_risk_data = pd.read_csv(
    "LUAD_low_risk_patients.csv")
# LUSC Low Risk
lusc_low_risk_data = pd.read_csv(
    "LUSC_low_risk_patients.csv")

# Concatenate them together.
data = pd.concat([luad_data, lusc_data], axis=0)
data.reset_index(drop=True, inplace=True)

# We're adding a new column as risk_of_patient with full of NaN
data['risk_of_patient'] = pd.Series(np.nan, index=data.index)

# Concatenate high risks for LUAD and LUSC
high_risk_data = pd.concat([luad_high_risk_data, lusc_high_risk_data], axis=0)
high_risk_data.reset_index(drop=True, inplace=True)
# Concatenate low risks for LUAD and LUSC
low_risk_data = pd.concat([luad_low_risk_data, lusc_low_risk_data], axis=0)
low_risk_data.reset_index(drop=True, inplace=True)

# Here we add a new column named 'risk_of_patient' into dataset which represent risk of patients as 'high_risk' or 'low_risk'
for i in range(data.shape[0]):
    # For high_risk it will be 1
    if (high_risk_data.isin([data.iloc[i]['submitter_id']]).any().any()):
        data['risk_of_patient'][i] = 1
    # For low_risk it will be 0
    elif (low_risk_data.isin([data.iloc[i]['submitter_id']]).any().any()):
        data['risk_of_patient'][i] = 0
    # For unknown it will be 2
    else:
        data['risk_of_patient'][i] = 2

# If there is a 'not reported' value in data we will replace it with NaN
data = data.replace("not reported", np.nan)
previev_of_data = data
# Describe of data
print(data.describe())


# PREPROCESSING #

# Function for find an index of an item in dataFrame.
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos


# In days_to_death column values are NaN for 'alive' patient and there is just information about 'dead' patients how long they lived.
# If there is a case like that if patient vital status is dead but days_to_death is 0. We'll drop that index.
indexes = getIndexes(data, 0)
for i, j in indexes:
    if (j == 'days_to_death' and data["vital_status"][i] == 'dead'):
        data.drop([i], inplace=True)

# There are almost 60 patient which has no information about risk_of_patient column we'll drop these rows.
indexes = getIndexes(data, 2)
for i, j in indexes:
    if (j == 'risk_of_patient'):
        data.drop([i], inplace=True)

# We will drop the columns which are fully NaN.
dataset_size = data.shape[0]
sum_of_nan_data = data.isnull().sum()
for col in data.columns:
    if (sum_of_nan_data[col] == dataset_size):
        data.drop([col], axis=1, inplace=True)

data.reset_index(drop=True, inplace=True)

# Now we don't have a column with fully NaN.
# year_of_death column has 736 null values therefore we'll drop it.
data.drop(["year_of_death"], axis=1, inplace=True)

'''We can drop year_of_birth column because we already have days_to_birth which is more suitable for our algorithm'''
data.drop(["year_of_birth"], axis=1, inplace=True)

'''In 'state' column all the variables are same for all rows as 'released'. We'll drop it'''
data.drop(["state"], axis=1, inplace=True)

# Drop updated_datetime because there is no special information about patient status
data.drop(["updated_datetime"], axis=1, inplace=True)

# Now we will fill the numeric NaN variables with the mean of columns.
numeric_columns = {"age_at_diagnosis", "days_to_birth", "cigarettes_per_day", "years_smoked"}
# Import SimpleImputer for fillin NaN values with the column's mean.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# We will take the indexes of these columns and fill the NaN values with column's mean.
for i in numeric_columns:
    data.iloc[:, data.columns.get_loc(i):data.columns.get_loc(i) + 1] = data.iloc[:,
                                                                        data.columns.get_loc(i):data.columns.get_loc(
                                                                            i) + 1].values
    data.iloc[:, data.columns.get_loc(i):data.columns.get_loc(i) + 1] = imputer.fit_transform(
        data.iloc[:, data.columns.get_loc(i):data.columns.get_loc(i) + 1])

# Now if patient is alive we'll assume that patients will die at 90 years old. Therefore we'll calculate their days_to_death value
for i in range(data.shape[0]):
    if (data["vital_status"][i] == 'alive'):
        new_days_to_death = 32850 + data['days_to_birth'][i]
        data['days_to_death'][i] = new_days_to_death

# NaN values

# At the end we should remove the ids. Because there is nothing special for these ids. These are just random variables.
id_columns = {"submitter_id", "diagnosis_id", "exposure_id", "demographic_id", "treatment_id", "bcr_patient_barcode",
              "days_to_last_follow_up"}
data.drop(id_columns, axis=1, inplace=True)

print(data.isnull().sum())

'''Now we have just have NaN values in 'ethnicity' and 'race' columns. These columns are highly related with cancer genetic
we will keep them as unknown'''
data[['ethnicity', 'race', 'risk_of_patient']] = data[['ethnicity', 'race', 'risk_of_patient']].fillna(value='unknown')

# ENCODING #
# We should convert String values into float or integer values
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

'''We need to encode our String values into integer values by applying LabelEncoder and OneHotEncoder.
'''

# It seems that tissue_or_organ_of_origin and site_of_resection_or_biopsy are identical features. We will make sure and if they
# are identical we'll drop one of them
if (data['tissue_or_organ_of_origin'].equals(data['site_of_resection_or_biopsy'])):
    data.drop(["tissue_or_organ_of_origin"], axis=1, inplace=True)
# NaN values
print("NaN values in dataset")
print(data.isnull().sum())
# Our categorial columns are: ("primary_diagnosis","tissue_or_organ_of_origin","morphology","site_of_resection_or_biopsy","tumor_stage","ethnicity","race")
categorical_columns = ("primary_diagnosis", "morphology", "tumor_stage",
                       "site_of_resection_or_biopsy", "ethnicity", "race", "vital_status", "gender", "disease")
# print(data.isnull().sum())
for i in categorical_columns:
    column = data.iloc[:, data.columns.get_loc(i):data.columns.get_loc(i) + 1].values
    column[:, 0] = le.fit_transform(data.iloc[:, data.columns.get_loc(i):data.columns.get_loc(i) + 1].values.ravel())
    # After apply LabelEncoder now we'll apply OneHotEncoder to represent categorical values better.
    # We will keep the new column names in column_names(Will generate them automatically)
    column_names = list()
    for j in range(int(np.unique(column[:, 0]).max()) + 1):
        column_names.append(i + "{}".format(j))
    ohe = preprocessing.OneHotEncoder()
    column = ohe.fit_transform(column).toarray()
    column_df = pd.DataFrame(data=column, columns=column_names)
    data.drop([i], axis=1, inplace=True)
    # There may be some dropped rows before therefore we reset the indexes.
    data.reset_index(drop=True, inplace=True)
    # Concatenate the new columns with dataset
    data = pd.concat([data, column_df], axis=1)

# APPLYING MACHINE LEARNING ALGORITHMS#

print(data.shape)

# Our ground truth vector will be risk_of_patient
y = data.risk_of_patient
X = data.drop(["risk_of_patient"], axis=1, inplace=False)

from sklearn.decomposition import PCA

# We split the data into train(%80) and test(%20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

machine_learning_algorithms = (SVC(), LogisticRegression(), KNeighborsClassifier(),
                               GaussianNB(), RandomForestClassifier(n_estimators=5))

ml_names = ("SVC", "Logistic Regression", "KNN", "Naive Bayes", "RandomForest")
for ml, ml_name in zip(machine_learning_algorithms, ml_names):
    # We split the data into train(%80) and test(%20)
    clf = ml
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print("{} Accuracy: %".format(ml_name), 100 - mean_absolute_error(y_test, predict) * 100)

# Changing of accuracy with PCA value
pca_parameters = (10, 20, 30, 40, 50, 60)
svc = []
logistic = []
knn = []
nb = []
rf = []
list_of_lists = (svc, logistic, knn, nb, rf)
list_index = 0
for ml in machine_learning_algorithms:
    for i in pca_parameters:
        pca = PCA(n_components=i)
        pca.fit(X)
        new_X = pca.transform(X)
        # We split the data into train(%80) and test(%20)
        X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=0)
        clf = ml
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        list_of_lists[list_index].append(100 - mean_absolute_error(y_test, predict) * 100)
    list_index += 1

# For KNN we can try to find best k_neighbours from 5 to 50.
knn_acuracies = []
for i in range(1, 50):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    knn_acuracies.append(100 - mean_absolute_error(y_test, predict) * 100)

print("For KNN algorithm the best n_neighbors is", knn_acuracies.index(max(knn_acuracies)) + 1, "with %",
      max(knn_acuracies), " accuracy score")

knn_acuracies = []
for i in range(1, 50):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    knn_acuracies.append(100 - mean_absolute_error(y_test, predict) * 100)

print("For RandomForest algorithm the best n_estimators is", (knn_acuracies.index(max(knn_acuracies)) + 1), "with %",
      max(knn_acuracies), " accuracy score")

