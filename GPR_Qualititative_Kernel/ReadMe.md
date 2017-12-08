### Qualitative Kernels for use with
### scikit-learn Gaussian Process Regressor

Reference:

A Simple Approach to Emulation for Computer Models With 
Qualitative and Quantitative Factors

Qiang Zhou, Peter Z. G. Qian & Shiyu Zhou

Technometrics Vol. 53 , Iss. 3,2011

<http://amstat.tandfonline.com/doi/full/10.1198/TECH.2011.10025>

Abstract
We propose a flexible yet computationally efficient approach for building 
Gaussian process models for computer experiments with both qualitative and 
quantitative factors. 
This approach uses the hypersphere parameterization to model the correlations 
of the qualitative factors, thus avoiding the need of directly solving 
optimization problems with positive definite constraints. 
The effectiveness of the proposed method is successfully illustrated by 
several examples.

The kernels proposed in this paper are implemented for use with the 
Gaussian Process Regressor in scikit-learn.  

Running P1B1/gprMBO_P1B1.py gives a minimal demonstration of their efficacy 
when applied to data from P1B1.  This will fit a GPR model to cached data,
and numerically optimize the resulting prediction function
(adding penalties to enforce feasabilty of the solution).  

Overall, it is patterned on mlrMBO, but removes the use of  
Random Forest Regression as a surrogate, instead directly modeling the 
qualitative factors in GPR. The Focus algorithm and Lower Confidence Bound 
search from mlrMBO are implemented as well.

Results from small test run
(3 smallest predicted values chosen from each method):

|	|Validation Loss|
|---------------------------|-------------------:|
|Optimization	|0.014496259291966756|
|Optimization	|0.2548767030239105|
|Optimization	|0.2533208906650543|
|Lower Confidence Bound	|0.025099394271771113|
|Lower Confidence Bound	|0.6990377942721049|
|Lower Confidence Bound	|NaN|

Compare to training data:

|   |validation_loss|
|-------|---------:|
|count    |95.000000|
|mean     | 0.694925|
|std      | 0.047132|
|min      | 0.664781|
|25%      | 0.671586|
|50%      | 0.682936|
|75%      | 0.698732|
|max      | 0.985598|