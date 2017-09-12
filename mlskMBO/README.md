Model Based Optimization using Python and scikit-learn machine learning instead of R. 

Short term goal: replace Random Forest Regression with Gaussian Process Regression as the model, even when there are categorical predictors. 

Long term: functional equivalence to mlrMBO.
ParameterSet object modelled on R version, sampling, Lower Confidence Bound, and focus are implemented.

Creates Python dictionaries for NT3 and optionally runs them in Keras.

Any predictor in scikit-learn for which standard errors are available should be fair game.
