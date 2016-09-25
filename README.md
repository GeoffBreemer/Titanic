# Project goal
The purpose of this project was to learn Python and experiment with scikit-learn and its Pipeline, FeatureUnion and other classes using Kaggle's Titanic competition. Achieving a high Kaggle score was not a goal. Code uses Python 3.5 and scikit-learn 0.17.1. All data files are located in the `data` subfolder

# Fitting the model
Run `driver_fit.py` to fit the RandomForest model, which will be saved to subfolder `model`

# Making predictions
Run `driver_predict.py` to make test set predictions and prepare the Kaggle submission file. It reads the RandomForest model created by `driver_fit.py` 

# Acknowledgments
New features are created based partially on code discussed on [this](http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html) web site
