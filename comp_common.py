'''Common functions and constants for Part 2 of the project'''
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------- Constants --------------------------
traindatafilename = './data/train.csv'
testdatafilename = './data/test.csv'
modelfilename = './model/titanicmodel.pkl'
startingColumn = 'Pclass'
labelColumn = 'Survived'

# -------------------------- Generic functions --------------------------
def loadData(filename):
    '''Load data'''
    return pd.read_csv(filename, sep=",", index_col='PassengerId', quotechar='"')


def writeData(filename, X, yPredict):
    '''Write predictions to file'''
    yPredict.to_csv(filename)

# -------------------------- Pipeline --------------------------

class PrepareCatFeatures(BaseEstimator, TransformerMixin):
    '''Convert strings to number. Could be replaced with a DictVectorizer'''

    def __init__(self):
        pass

    def transform(self, X, y=None):
        # Suppress SettingWithCopyWarning (alternatively: add a X = X.copy()
        with pd.option_context('mode.chained_assignment', None):
            # --- Convert Embarked
            mapping = {'S': 0,
                       'C': 1,
                       'Q': 2,
                       }
            X.loc[:, 'Embarked'] = X.loc[:, 'Embarked'].replace(mapping, inplace=False)

            # --- Convert Sex
            mapping = {'female': 0,
                       'male': 1
                       }
            X.loc[:, 'Sex'] = X['Sex'].replace(mapping, inplace=False)

            # --- Convert Name to Title
            X.loc[:, 'Title'] = X['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

            # a map of more aggregated titles
            mapping = {
                "Capt": 0,  # Officer
                "Col": 0,  # Officer
                "Major": 0,  # Officer
                "Jonkheer": 1,  # Royalty
                "Don": 1,  # Royalty
                "Sir": 1,  # Royalty
                "Dr": 0,  # Officer
                "Rev": 0,  # Officer
                "the Countess": 1,  # Royalty
                "Dona": 1,  # Royalty
                "Mme": 2,  # "Mrs"
                "Mlle": 3,  # "Miss"
                "Ms": 2,  # "Mrs"
                "Mr": 4,  # "Mr"
                "Mrs": 2,  # "Mrs"
                "Miss": 3,  # "Miss"
                "Master": 5,  # "Master"
                "Lady": 1  # "Royalty"
            }
            X.loc[:, 'Title'] = X['Title'].map(mapping)

        X = X.drop('Name', 1)
        return X

    def fit(self, X, y=None):
        return self


def dropUnusedColumns(X):
    return X.drop('Cabin', 1)


def selectContFeatures(X):
    return X[['Fare', 'Parch', 'SibSp']]


def selectCatFeatures(X):
    return X[['Pclass', 'Sex', 'Embarked', 'Name']]


def oneHotFeatures(X):
    return X[:, [3, 4, 5, 6, 9]]


def notOneHotFeatures(X):
    return X[:, [0, 1, 2, 7, 8]]


def selectPassThrough(X):
    return X[['Age']]


def imputeAge(X):
    '''Impute age for NaNs'''
    def fillAges(row, sexCol, titleCol, pclassCol, ageCol):

        if not(np.isnan(row[-1])):
            return row

        if row[sexCol] == 0 and row[pclassCol] == 1:
            if row[titleCol] == 3:
                row[ageCol] = 30
            elif row[titleCol] == 2:
                row[ageCol] = 45
            elif row[titleCol] == 0:
                row[ageCol] = 49
            elif row[titleCol] == 1:
                row[ageCol] = 39

        elif row[sexCol] == 0 and row[pclassCol] == 2:
            if row[titleCol] == 3:
                row[ageCol] = 20
            elif row[titleCol] == 2:
                row[ageCol] = 30

        elif row[sexCol] == 0 and row[pclassCol] == 3:
            if row[titleCol] == 3:
                row[ageCol] = 18
            elif row[titleCol] == 2:
                row[ageCol] = 31

        elif row[sexCol]  == 1 and row[pclassCol] == 1:
            if row[titleCol] == 5:
                row[ageCol] = 6
            elif row[titleCol] == 4:
                row[ageCol] = 41.5
            elif row[titleCol] == 0:
                row[ageCol] = 52
            elif row[titleCol] == 1:
                row[ageCol] = 40

        elif row[sexCol] == 1 and row[pclassCol] == 2:
            if row[titleCol] == 5:
                row[ageCol] = 2
            elif row[titleCol] == 4:
                row[ageCol] = 30
            elif row[titleCol] == 1:
                row[ageCol] = 41.5

        elif row[sexCol] == 1 and row[pclassCol] == 3:
            if row[titleCol] == 5:
                row[ageCol] = 6
            elif row[titleCol] == 4:
                row[ageCol] = 26

        return row

    pclassIx = 3
    titleIX = 6
    sexIx = 4
    ageCol = 7

    X = np.apply_along_axis(fillAges, 1, X, sexIx, titleIX, pclassIx, ageCol)

    return X


def addFamily(X):
    # Family size: index 8
    newCol = np.array(X[:, 1] + X[:, 2], np.newaxis)
    newCol = newCol.reshape((len(newCol), 1))
    X = np.hstack( (X,newCol) )

    # Family category: index 9
    def determineFamilyCat(row):
        # print('row shape = {}, cont = {}'.format(row.shape, row))
        if row[8] == 1:
            return 0   # singles
        elif 2<=row[8]<=4:
            return 1   # normal size
        else:
            return 2   # large size

    newCol = np.apply_along_axis(determineFamilyCat, 1, X)
    newCol = newCol.reshape((len(newCol), 1))
    X = np.hstack((X,newCol))

    return X


# Not used
class PrepDictVect(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer, return as a dict"""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, featMatrix):
        return [{'Pclass': int(row[0]),
                 'Sex': row[1],
                 'Embarked': row[2],
                 'Title': row[3],
                 'Famcat': row[4]
                 }
                for row in featMatrix]
