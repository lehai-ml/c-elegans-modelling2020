import sys
import genetic_algorithm as gen
import pandas as pd
import time

def main():
    
    dataset=pd.read_excel('./dataset.xlsx')

    # by_15C=dataset[dataset['temperature']==15]
    by_20C=dataset[dataset['temperature']==20]
    # by_25C=dataset[dataset['temperature']==25]
    
    PSO=input('use PSO= ')
    verbose=input('use verbose= ')
    population_size=int(input('population size= '))
    n_generation=int(input('n generation= '))
    n_parents=int(input('n_parents= '))
    offspring_size=int(input('offspring size= '))
    file=input('filename is ')
    start=time.time()
    with open(file,'wb') as output:
        output.write('use PSO= ',PSO)
        output.write('population size= ',population_size)
        output.write('n_generation= ',population_size)
        output.write('n_parents= ',population_size)
        output.write('offspring_size= ',population_size)
    test=gen.GA(population_size=population_size,
                n_generation=n_generation,
                n_parents=n_parents,
                offspring_size=offspring_size,
                df_temp_food=by_20C)
    population,population_connection=test.running_GA(PSO=PSO,verbose=verbose,file=file)
    end=time.time()
    print(end-start)
    with open(file,'a') as output:
        output.write('total running time',(end-start))
    
    
if __name__=='__main__':
    main()


# def main():
#     # print command line arguments
#     print(sys.argv[1])

# if __name__ == "__main__":
#     main()
    
# ##sanity check #Version1 model:
# neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1,DA2S=-1,TA2S=-1)
# neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=1,DA2S=-1)
# neuron_ASI=elegans.define_model_interactions('asi',TN2S=1,DA2S=1,TA2S=1)
# model1=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
# Version1=elegans.Model(model1,by_20C)
# Version1.model['nsm']

# ##Simulation with odeint
# print(Version1.DA_table)




# ##Version3 model Waleed paper:
# ##sanity check #Version3 model (Waleed thesis):
# neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1,DA2S=-1)
# neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=-1)
# neuron_ASI=elegans.define_model_interactions('asi',DA2S=1,TA2S=-1)
# model3=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
# Version3=elegans.Model(model3,by_20C)


# neuron_NSM=elegans.define_model_interactions('nsm',TN2S=-1)
# neuron_ADF=elegans.define_model_interactions('adf',TA2S=1,TN2S=-1)
# neuron_ASI=elegans.define_model_interactions('asi',DA2S=1,TA2S=-1)
# model3=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})




