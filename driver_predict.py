from sklearn.externals import joblib
from comp_common import *

# -------------------------- Constants --------------------------
predfilename = './data/testpredictions.csv'

# -------------------------- DRIVER CODE STARTS HERE --------------------------


# -------------------------- Load the data set --------------------------
print('---> loading data')
X = loadData(testdatafilename)

# -------------------------- Load the model --------------------------
print('---> loading model')
clf = joblib.load(modelfilename)

# -------------------------- Write test predictions to file --------------------------
yPredict = pd.DataFrame(clf.predict(X))

yPredict.columns = ['Survived']
yPredict.index = X.index

writeData(predfilename, X, yPredict)

print('---> predictions complete')
