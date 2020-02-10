# C. elegans modelling
## Objective
Given fluorescence level of 2 genes (tph-1 and daf-7) in 3 neurons across 6 different food levels, 3 temperatures and 4 genetic backgrounds, the project aimed to fit interactions between those neurons and the effect of food and temperature.
## Materials and methods
dataset.xlsx- supplementary information from [Entchev et al., 2015](https://elifesciences.org/articles/06259)

Script requires sympy, numpy, pandas, matplotlib and scipy.
## Updates
**10.02.2020- (See elegansfunc.py and main notebook)**
* (See simulation_in_tph1mut/daf7/WT)- updated simulation function. Utilizes scipy.integrate.odeint library

**07.02.2020- (See elegansfunc.py)**
* (See define_model_interactions and write_equation )Hard-coded calculation of DA, TA and TN for any given model interactions. Models are restricted to 3 neurons. An examined neuron can receive up to 3 DIRECT connections to its cell body (i.e. 2 from the other 2 unexamined neurons, and 1 self-regulatory(i.e. from itself)). The self-regulatory can receive 1 further connections from each of the other two cells. The other 2 DIRECT connections can receive 1 more connection each from the opposite cell (not examined one).
* (See write_equation) From these defined interactions, for each neuron is generated a differential equation.
* (See find_DA_at_food_level) calculates DA (daf-7 effect) from tph-1 mutants and double mutants
* (See find_TA_TN_in_daf7mut) calculates TA/TN (tph-1 effect) from daf-7 mutants and wild-types