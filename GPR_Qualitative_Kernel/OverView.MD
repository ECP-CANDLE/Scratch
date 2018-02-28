Hyperparameter Optimization
mlrMBO uses Gaussian Process Regression with a Radial Basis Function (also known as Squared Euclidean) as the kernel when all variables are continuous.  When categorical factors are added, it proposes instead to use Random Forest Regression (or any other predictor with a standard error) as a surrogate.  This approach is problematic, as has been noted by the authors of mlrMBO themselves:  
<https://mlr-org.github.io/mlrMBO/articles/supplementary/mixed_space_optimization.html#remarks>

They comment that

>For the random forest the uncertainty is based on the diversity of the results of the individual individual forests.

However, this should be amplified to note that the construction of the forest itself will seek to minimize this diversity.  In practice, the Random Forest in mlrMBO will tend to identify and explore near the best value observed so far.  (This behavior has been explored in some detail with the aid of a Python/scikit-learn mockup.)

However, one major advantage of Gaussian Process Regression is that hyperparameters are included in fitting the model itself.  This motivates finding an alternative that can accomodate the categorical variables as part of the model.  A simple approach is simply to add dummy-coded indicator variables to an RBF kernel, sometimes described as Automatic Relevance Detection, or ARD.  If the associated length-scale parameter is large, the parameter is essentially ignored in the model.
This approach was tested with success for the NT3 model, as shown in [mlskMBO (replacing R with scikit-learn in MBO](../tree/master/mlskMBO).

A much richer model for categorical factors has been proposed by Zhou, Quan and Zhou <link>. This model uses a parameterization of the hypersphere to model correlations between different levels of a factor, and has been implemented in Python.  It is fully integrated into the scikit-learn Gaussian Process Regression.  The most time-consuming portions of the calculation have been isolated and implemented in Cython, [hypersphere](../hypershpere) allowing parallel computation using OpenMP when available.
