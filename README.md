# C. elegans modelling
## Objective
Given fluorescence level of 2 genes (tph-1 and daf-7) in 3 neurons across 6 different food levels, 3 temperatures and 4 genetic backgrounds, the project aimed to fit interactions between those neurons and the effect of food and temperature.
## Materials and methods
dataset.xlsx- supplementary information from [Entchev et al., 2015](https://elifesciences.org/articles/06259)

Script requires sympy v. 1.1.1, numpy 1.15.4, pandas 0.23.0, matplotlib 2.2.2 and scipy 1.1.0./ Python v.3.6.8

## Updates
**17.02.2020**
* Constrained solutions to real numbers in sympy. If there is no solution for real number, optimization with PSO is employed (applies to all parameters). particleswarmop is for TA and TN estimation, and particleswarmop_forDA is for DA estimation.
* Genetic algorithm runs roughly 10 minutes for population_size=3,n_generation=3,n_parents=2,offspring_size=1, with the best final model MSE of 0.016.
* Added regularization term for calculation of MSE to penalize for bigger models (too many connections).


**15.02.2020- (See genetic_algorithm.py)**
* Implemented Genetic Algorithm to identify the best model based on mean score error.

**14.02.2020- (See particleswarmop.py)**
* Implemented Particle Swarm Optimization. It seems to work well, although takes almost double time to compute compared with sympy library (although we could decrease the n_iteration and increase target_error). The advantage is that we have more control over the parameter search.
Mean squared error on both methods appear to be very similar (PSO : sympy= 0.029 : 0.031, on Version1 with n_iteration=100, and target_error=10e-6)

**10.02.2020- (See elegansfunc.py and main notebook)**
* (See simulation_in_tph1mut/daf7/WT)- updated simulation function. Utilizes scipy.integrate.odeint library

**07.02.2020- (See elegansfunc.py)**
* (See define_model_interactions and write_equation )Hard-coded calculation of DA, TA and TN for any given model interactions. Models are restricted to 3 neurons. An examined neuron can receive up to 3 DIRECT connections to its cell body (i.e. 2 from the other 2 unexamined neurons, and 1 self-regulatory(i.e. from itself)). The self-regulatory can receive 1 further connections from each of the other two cells. The other 2 DIRECT connections can receive 1 more connection each from the opposite cell (not examined one).
* (See write_equation) From these defined interactions, for each neuron is generated a differential equation.
* (See find_DA_at_food_level) calculates DA (daf-7 effect) from tph-1 mutants and double mutants
* (See find_TA_TN_in_daf7mut) calculates TA/TN (tph-1 effect) from daf-7 mutants and wild-types