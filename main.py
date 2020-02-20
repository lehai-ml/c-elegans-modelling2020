import sys
import genetic_algorithm as gen
import pandas as pd
import time

def main():
    
    dataset=pd.read_excel('./dataset.xlsx')
    
    by_20C=dataset[dataset['temperature']==20]
    
    PSO=bool(input('use PSO= '))
    verbose=bool(input('use verbose= '))
    population_size=int(input('population size= '))
    n_generation=int(input('n generation= '))
    n_parents=int(input('n_parents= '))
    offspring_size=int(input('offspring size= '))
    file=input('filename is ')
    start=time.time()
    with open(file,'w') as output:
        if PSO==True:
            output.write('using PSO'+"\n")
        output.write('population size= '+str(population_size)+"\n")
        output.write('n_generation= '+str(n_generation)+"\n")
        output.write('n_parents= '+str(n_parents)+"\n")
        output.write('offspring_size= '+str(offspring_size)+"\n")
    test=gen.GA(population_size=population_size,
                n_generation=n_generation,
                n_parents=n_parents,
                offspring_size=offspring_size,
                df_temp_food=by_20C)
    
    population,population_connection=test.running_GA(PSO=PSO,verbose=verbose,file=file)
    end=time.time()
    print(end-start)
    with open(file,'a') as output:
        output.write('total running time'+str(end-start)+"\n")
    
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




