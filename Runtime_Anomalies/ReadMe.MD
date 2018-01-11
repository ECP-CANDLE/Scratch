### Run Time Anomalies in P1B1

The file `suspect.json` contains half a dozen parameter dictionaries 
for p1b1_baseline_keras2 which have the longest run times.
As it turns out, they also have the longest run time per epoch.

The first parameter dictionary has an extravagantly large 
validation loss.  The next two have long run times, but small
validation losses.  The fourth and fifth have very typical values
for the loss, while the last has a large loss, though much smaller
than the first:

validation_loss | runtime_hours | epochs | hrs_per_epoch
----------------|---------------|-------:|---------------
44.0339033 | 0.133340106667 | 169 | 0.000788994714004
 0.205439427892 | 0.125816531944 | 166 | 0.000757930915328
 0.251138623556 | 0.147469485556 | 195 | 0.00075625377208
 0.671716454824 | 0.0808523552778 | 108 | 0.000748632919239
 0.671307932536 | 0.0967377202778 | 133 | 0.000727351280284
 3.68405342738 | 0.121794521944 | 169 | 0.00072067764

#### Setup

Clone Scratch to a directory at the same level as Benchmarks:

```
 GitHub
    Scratch
        Runtime_Anomalies
    Benchmarks
        common
        Pilot1
            common
            P1B1
```
            
If located elsewhere, edit p1b1_paths.py as needed, or ensure that  
p1b1_baseline_keras2 is on the PYTHONPATH.

Assuming P1B1 has all dependencies installed (especially keras!), simply run:

`python p1b1_runtime_anomalies.py`

In each of the six .json log files which should result, the parameters used  
may be found near the beginning.  The final validation_loss and runtime_hours  
will be written at the end of the log file.

There are comments in the code which describe how to change the save directory,  
or rename the logfile.
