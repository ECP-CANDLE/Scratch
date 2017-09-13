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
