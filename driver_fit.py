# Feature selection is based on:
#
# http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html


# -------------------------- Imports --------------------------
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, Imputer, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from comp_common import *

# -------------------------- Constants --------------------------
randomState = 122177
numCVFolds = 10
testSize = 0.2
kRange = np.arange(15, 0, -1)
weightRange = ['uniform', 'distance']
setSizes = np.linspace(0.1, 1.0, 20)
savePlots = False
numFeatures = 5
scoring = 'accuracy'

# -------------------------- DRIVER CODE STARTS HERE --------------------------

# -------------------------- Load the data set --------------------------
print('---> loading data')
data = loadData(traindatafilename)

# Split into a feature matrix and label vector
X = data.loc[:, startingColumn:]
y = data.loc[:, [labelColumn]]
y = y.values.ravel()            # to suppress DataConversionWarning warnings


# -------------------------- Split the FULL data set in a TRAIN and TEST split --------------------------
print('---> splitting data set')
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=randomState)

# -------------------------- Tune hyper parameters for the selected model using CV on the TRAIN data set --------------------------
print('---> hyper parameter tuning: grid search')

# Feature preparation for continuous variables
continuousPipeline = Pipeline([
    ('selcont', FunctionTransformer(selectContFeatures, validate=False)),
    ('impu',    Imputer(missing_values="NaN", strategy="median")),
])      # 'Fare', 'Parch', 'SibSp' with NaNs replaced with the median

# Feature preparation for categorical variables
categoricalPipeline = Pipeline([
    ('selcat',  FunctionTransformer(selectCatFeatures, validate=False)),
    ('prep',    PrepareCatFeatures()),     # ['Pclass', 'Sex', 'Embarked', 'Name'] -> Name renamed Title
    ('impu',    Imputer(missing_values="NaN", strategy="most_frequent")),   # only for integers
])      # 'Pclass', 'Sex', 'Embarked', 'Title'

# Pass through 'Age'
passThroughPipeline = Pipeline([
    ('select',  FunctionTransformer(selectPassThrough, validate=False)),
])      # 'Age', no changes

# Perform continuous and categorical feature preparation in parallel
prepareFeaturesParallel = FeatureUnion([
    ('cont',    continuousPipeline),
    ('cat',     categoricalPipeline),
    ('age',     passThroughPipeline)
])

# Perform one-hot-encoding for cat variables, scale cont variables
oneHotPipeline = Pipeline([
    ('selone',  FunctionTransformer(oneHotFeatures, validate=False)),
    # ('prep', PrepDictVect()),                 # Only works on string features
    # ('dict', DictVectorizer(sparse=False))    # Features are already integers, so DictVectorizer won't do anything
    ('hot',     OneHotEncoder(sparse=False)),   # OneHotEncoder works on integers
])

scalePipeline = Pipeline([
    ('selnot',  FunctionTransformer(notOneHotFeatures, validate=False)),
    ('exec',    MinMaxScaler())
])

oneHotAndScalePipeline = FeatureUnion([
    ('onehot',  oneHotPipeline),
    ('exec',    scalePipeline)
])

featurePreproc = Pipeline([
    ('all',     prepareFeaturesParallel),
    ('age',     FunctionTransformer(imputeAge, validate=False)),
    ('family',  FunctionTransformer(addFamily, validate=False)),
    ('hot',     oneHotAndScalePipeline),
])

# Final column indices:
# fare    parch   sibs    pclass  sex embarked    title   age famsize   famcat
# 0       1       2       3       4   5           6       7   8         9
# one-hot?
# NO      NO      NO      YES     YES YES         YES     NO  NO        YES


# Run a pipeline without the classifier to inspect intermediate results:
# lala = featurePreproc.fit_transform(XTrain)
# lala = prepareFeaturesParallel.fit_transform(XTrain)
# print(lala.shape)
# print(type(lala))
# print(lala[0,:])
# exit()

# Create the main pipeline with the classifier as the last step
pipeline = Pipeline(steps=[
    ('drop', FunctionTransformer(dropUnusedColumns, validate=False)),
    ('fpre', featurePreproc),
    ('pca', PCA(n_components=6)),
    ('clf', RandomForestClassifier(n_jobs=-1, random_state=randomState))
])

# kfold = KFold(n=len(XTrain), n_folds=numCVFolds, random_state=randomState, shuffle=True)
kfold = StratifiedKFold(yTrain, n_folds=numCVFolds, random_state=randomState, shuffle=True)
param_grid = dict(
    clf__max_depth=[3, 5, 10],
    clf__min_samples_split=[2, 10, 20],
    clf__min_samples_leaf=[1, 5, 10],
)
grid = GridSearchCV(pipeline, cv=kfold, param_grid=param_grid, scoring=scoring)
grid.fit(XTrain, yTrain)

print('                     Best parameters: {}'.format(grid.best_params_))
print('                         CV accuracy: {:7.4f}'.format(grid.best_score_))


# -------------------------- Obtain final performance estimate for unseen data by making predictions on the TEST data set --------------------------
print('---> estimate performance on unseen data')
yPredTest = grid.predict(XTest)
print('Estimated performance on unseen data: {:7.4f}'.format(accuracy_score(yTest, yPredTest)))
# print(classification_report(yTest, yPredTest))


# -------------------------- Fit final model on the FULL data set (i.e. TRAIN and TEST combined) --------------------------
print('---> fitting final model')
grid.fit(X, y)


# -------------------------- Save the pipeline using pickle/joblib --------------------------
print('---> saving final model')
joblib.dump(grid, modelfilename)

print('---> fitting complete')
