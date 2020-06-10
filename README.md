# Sporadic CCM

This package contains the python implementation of the paper : "Inferring Causal Dependencies between Chaotic Dynamical Systemsfrom Sporadic Time Series"

## Dependencies.

gru_ode : https://github.com/edebrouwer/gru_ode_bayes
skccm : https://skccm.readthedocs.io/en/latest/

## Installation.
From the top directory, run :
```
pip install -e . 
```

## Running Code

### Data Generation
Generation of sporadic double pendulum trajectories is computed using : 
````
python data_generation_script.py
````

### Filtering of the sporadic time series
The following script will train a GRU-ODE-Bayes filtering model on top of the given data and reconstruct the full trajectory accordingly.
```
.\launch_gru_ode.sh
```
Trained models are saved in the trained_models folder.

### Causal direction inference
We can then compute the scores for causal dependence between dynamic systems : 
```
python gruode_scores.py
```
The scores are saved in results_ccm.csv




