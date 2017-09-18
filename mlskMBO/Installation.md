Clone Benchmarks, then clone Scratch alongside it in the same directory, and mlskMBO will be able to import NT3_baseline_keras2 and common files (or P3B1, etc.).  Alternatively, customize the list of paths in mlskMBO.py to point to the actual location of Benchmarks and Pilot1, or use PYTHONPATH.
 
The code in nt3_run_data scrapes the json log files and collects the results into a pandas dataframe.   

Currently, results from an initial grid with a few hundred points are stored in nt3_initial_data.csv and read into a dataframe, to streamline demonstration.  The get_nt3_data function can be directed to the location of json logs from NT3 if available.

A Gaussian Process Regression models the response (validation loss), and scikit-learn 'optimize' finds the location of the minimal value of the model.  This is used as the starting point for the focus algorithm, which creates dictionaries of parameter values.  The resulting list of dictionaries may optionally be evaluated by nt3_baseline_keras2.

The json log files which are produced could then be read in to define a new model, and so on ... 
