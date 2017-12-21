### Qualitative Kernels for use with
### scikit-learn Gaussian Process Regressor

#### Reference:

A Simple Approach to Emulation for Computer Models With 
Qualitative and Quantitative Factors

Qiang Zhou, Peter Z. G. Qian & Shiyu Zhou

Technometrics Vol. 53 , Iss. 3,2011

<http://amstat.tandfonline.com/doi/full/10.1198/TECH.2011.10025>

#### Abstract
> We propose a flexible yet computationally efficient approach for building 
> Gaussian process models for computer experiments with both qualitative and 
> quantitative factors. 
> This approach uses the hypersphere parameterization to model the correlations 
> of the qualitative factors, thus avoiding the need of directly solving 
> optimization problems with positive definite constraints. 
> The effectiveness of the proposed method is successfully illustrated by 
> several examples.

The kernels proposed in the referenced paper are implemented for use with the 
Gaussian Process Regressor in scikit-learn.  Some of the most expensive numerical
computations in the gradient computation are accelerated by compiling with cython.

Running P1B1/gprMBO_P1B1.py gives a minimal demonstration of their efficacy 
when applied to data from P1B1.  This will fit a GPR model to cached data,
and numerically optimize the resulting prediction function
(adding penalties to enforce feasibility of the solution).  

NT3/gprMBO_NT3.py is similar, but also demonstrates sampling from a grid to 
obtain initial data, running keras, recovering the results, then fitting the
GPR model, optimizing, and adding the results to the initial data.

Overall, it is patterned on mlrMBO, but removes the use of  
Random Forest Regression as a surrogate, instead directly modeling the 
qualitative factors in GPR. The Focus algorithm and Lower Confidence Bound 
search from mlrMBO are implemented as well.

Results from small test run
(3 smallest predicted values chosen from each method):

|Method	|Validation Loss|
|---------------------------|-------------------:|
|Optimization	|0.014496259291966756|
|Optimization	|0.2548767030239105|
|Optimization	|0.2533208906650543|
|Lower Confidence Bound	|0.025099394271771113|
|Lower Confidence Bound	|0.6990377942721049|
|Lower Confidence Bound	|NaN|

Compare to training data:

|   |validation_loss|
|--------|---------:|
|count    |95.000000|
|mean     | 0.694925|
|std      | 0.047132|
|min      | 0.664781|
|25%      | 0.671586|
|50%      | 0.682936|
|75%      | 0.698732|
|max      | 0.985598|

An earlier test run was even more successful:

|Method	|Validation Loss|
|---------------------------|-------------------:|
|Optimization	|0.025792453487714133|
|Optimization	|0.027282313605149586|
|Optimization	|0.013135291449725629|
|Lower Confidence Bound	|0.190620556473732|
|Lower Confidence Bound	|0.014625415603319803|
|Lower Confidence Bound	|0.007932033731291692|

#### Installation
Requires numpy, scipy, scikit-learn, pandas, and optionally, cython.  A standard Anaconda 
installation will have everything except cython; `conda install cython` should take care of that.
(If cython is not available, it will use a pure python version.)

Clone Scratch alongside Benchmarks, or edit paths near the beginning of the demo to match the 
actual directory structure.

If cython is installed, in the directory containing hypersphere_cython.pyx, run:
`python setup.py build_ext --inplace`

There is a "Configuration" section near the beginning of each demo.  `run_keras = True` will 
submit the parameter dictionaries to keras.  This is currently `False` for the P1B1 demo, to 
show just the GPR model and optimization.

A P3B1 demo should be available soon.  This will show how the Affinity Propagation clustering 
is used to identify the distinct local minima (corresponding to the 5 sub-problems from which
the average loss was constructed).
