Model Based Optimization using Python and scikit-learn machine learning instead of R. 

Short term goal: replace Random Forest Regression with Gaussian Process Regression as the model, even when there are categorical predictors. 

Long term: functional equivalence to mlrMBO.
ParameterSet object modelled on R version, sampling, Lower Confidence Bound, and focus are implemented.

Creates Python dictionaries for NT3 and optionally runs them in Keras.

Any predictor in scikit-learn for which standard errors are available should be fair game.


Focus Algorithm is an object-oriented Python adaptation of Algorithm 1 from the mlrMBO paper:

    Algorithm 1 Infill Optimization: Focus Search.
    Require: infill criterion c : X → R, control parameters nrestart, niters, npoints
    for u∈{1,...,nrestart} do 
        Set X ̃ = X
        for v ∈ {1,...,niters} do
            generate random design D ⊂ X ̃ of size npoints
            compute x∗u,v = (x∗1 , ..., x∗d ) = arg minx∈D c(x)
            shrink X ̃ by focusing on x∗:
            for each search space dimension X ̃ in X ̃ do
                if X ̃ numeric: X ̃ = [li, ui] then
                    li=max{li,x∗i − (ui−li)/4}
                    ui=min{ui,x∗i + (ui−li)/4}
                end if
                if X ̃ categorical: X ̃ = {vi1 ,...,vis }, s > 2 then
                    x ̄ = sample one category uniformly from X ̃ \x∗
                    X ̃i = X ̃i \ x ̄i
                end if 
            end for
        end for
    end for
    Return x∗ = argmin c(x∗u,v) u∈{1,...,nrestart },v∈{1,...,niters }


[mlrMBO: A Modular Framework for Model-Based Optimization of Expensive Black-Box Functions
Bernd Bischla, Jakob Richterb, Jakob Bossekc, Daniel Hornb, Janek Thomasa, Michel Langb]


mlskMBO.py currently is configured to gives a minimalistic demonstration.

The code in nt3_run_data scrapes the json log files produced by running nt3_baseline_keras2, and collects the results into a pandas dataframe.   

Currently, results from an initial grid with a few hundred points are stored in nt3_initial_data.csv and read into a dataframe, to streamline demonstration.  The get_nt3_data function can be directed to the location of json logs from NT3 if available.

A Gaussian Process Regression models the response (validation loss), and scikit-learn 'optimize' finds the location of the parameter values which give the minimal value of the model.  This is used as the starting point for the focus algorithm, which draws parameter values at random, but becoming closer to the starting point.  The resulting list of dictionaries may optionally be evaluated by nt3_baseline_keras2.

The json log files which are produced could then be read in with get_nt3_data to define a new model, and so on ... 
