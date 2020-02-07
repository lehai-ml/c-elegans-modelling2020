import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.solvers import solve
from sympy import symbols, Eq
import elegansfunc as elegans

dataset=pd.read_excel('./dataset.xlsx')

by_15C=dataset[dataset['temperature']==15]
by_20C=dataset[dataset['temperature']==20]
by_25C=dataset[dataset['temperature']==25]

##sanity check #Version1 model:
neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1,DA2S=-1,TA2S=-1)
neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=1,DA2S=-1)
neuron_ASI=elegans.define_model_interactions('asi',TN2S=1,DA2S=1,TA2S=1)
model1=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
Version1=elegans.Model(model1,by_20C)
Version1.model['nsm']

##Version3 model Waleed paper:
##sanity check #Version3 model (Waleed thesis):
neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1,DA2S=-1)
neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=-1)
neuron_ASI=elegans.define_model_interactions('asi',DA2S=1,TA2S=-1)
model3=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
Version3=elegans.Model(model3,by_20C)


neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1)
neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=-1)
neuron_ASI=elegans.define_model_interactions('asi',DA2S=1,TA2S=-1)
model3=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
Version4=elegans.Model(model3,by_20C)

# Version4.TA_TN_across_food_levels()
model3=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
Version3=elegans.Model(model3,by_20C)
Version1.model['nsm'].write_equation(print_equation=True)