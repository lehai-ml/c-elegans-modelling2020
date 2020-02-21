import sys
import genetic_algorithm as gen
import elegansfunc as elegans
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

def run_simulation():
    dataset=pd.read_excel('./dataset.xlsx')
    by_20C=dataset[dataset['temperature']==20]
    connections=dict()
    file=input('filename is ')
    PSO=bool(input('use PSO? (press enter for no)= '))
    verbose=bool(input('use verbose? (press enter for no)= '))
    gamma=float(input('gamma= '))
    normalize=bool(input('use normalize? (press enter for no)= '))
    for i in ['nsm','asi','adf']:
        connections[i]=list(int(num) for num in input("Enter the connections for "+i+" separated by comma: ").strip().split(','))[:7]
    neuron_NSM=elegans.define_model_interactions('nsm',TA2TN=connections['nsm'][0],TA2DA=connections['nsm'][1],TA2S=connections['nsm'][2],TN2S=connections['nsm'][3],DA2TA=connections['nsm'][4],DA2TN=connections['nsm'][5],DA2S=connections['nsm'][6])
    neuron_ADF=elegans.define_model_interactions('adf',TA2S=connections['adf'][0],TN2TA=connections['adf'][1],TN2DA=connections['adf'][2],TN2S=connections['adf'][3],DA2TA=connections['adf'][4],DA2TN=connections['adf'][5],DA2S=connections['adf'][6])
    neuron_ASI=elegans.define_model_interactions('asi',TA2TN=connections['asi'][0],TA2DA=connections['asi'][1],TA2S=connections['asi'][2],TN2TA=connections['asi'][3],TN2DA=connections['asi'][4],TN2S=connections['asi'][5],DA2S=connections['asi'][6])
    model1=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
    print(model1)
    if normalize==True:
        temp_model=elegans.Model(model1,elegans.normalize_by_highest_wildtype_mean(by_20C),PSO=PSO,verbose=verbose)
    else:
        temp_model=elegans.Model(model1,by_20C,PSO=PSO,verbose=verbose)
    print(temp_model.DA_table)
    print(temp_model.TA_TN_table)
    with open(file,'w') as output:
        if PSO ==True:
            output.write('using PSO'+"\n")
        if normalize==True:
            output.write('normalize is true'+"\n")
        output.write(str(gamma)+"\n")        
        output.write(str(connections)+"\n")
        output.write((temp_model.DA_table).to_string()+"\n")
        output.write((temp_model.TA_TN_table).to_string()+"\n")
    run_simulate=bool(input('run simulation? (press enter for no)= '))
    if run_simulate==True:
        test, experimental, mse_score=temp_model.simulate_all_together(gamma=gamma,compare=False)
        f,axes=plt.subplots(3,3,sharey=True,sharex=True,figsize=(15,10))
        neuron=['NSM','ADF','ASI']
        genotypes=['tph1mut','daf7mut','wildtype']
        for row in range(3):
            for column in range(3):
                ax=axes[row,column]
                genotype=genotypes[row]
                ax.plot(test[test['genotype']==genotype]['Food'],test[test['genotype']==genotype][neuron[column]],'rx--',label='simulated')
                ax.plot(experimental[experimental['genotype']==genotype]['Food'],experimental[experimental['genotype']==genotype][neuron[column]],'gx-',label='experimental')
                ax.plot(np.linspace(1e0,10e10,6),np.repeat(1,6),'c--')
                ax.set_xscale('log')
                ax.set_xticks([1e0,1e2,1e4,1e6,1e8,1e10])
                ax.set_xticklabels(['$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^10$'])
                ax.set_yticks(np.linspace(0.5,1.5,11))
                ax.tick_params(top=True,right=True,direction='in')
                if row==0:
                    ax.set_title(neuron[column],fontdict={'fontsize':20})
                if column==0:
                    ax.set_ylabel(genotype,fontsize=18)
                if ax==axes[2,2]:
                    handles, labels = ax.get_legend_handles_labels()
                    
        f.legend(handles, labels, loc='lower center',mode='expand',ncol=4,bbox_to_anchor=(0.24,0,0.5,0),
                    labelspacing=1.2,columnspacing=5,fontsize='x-large',frameon=False)
        f.text(0.5, 0.06, 'Food Levels (cells/ml)', ha='center',size='xx-large')
        f.text(0.05, 0.5, 'Normalised Fluorescent Expression AU', va='center', rotation='vertical',size='xx-large')
        f.suptitle('Normalized Fluorescent Expression by Food Level across neurons at 20 degree',size='xx-large')
        with open(file,'a') as output:
            output.write(str(mse_score)+"\n")
        plt.show()
        
        
        

def run_GA():
    
    dataset=pd.read_excel('./dataset.xlsx')
    by_20C=dataset[dataset['temperature']==20]
    PSO=bool(input('use PSO? (press enter for no)= '))
    verbose=bool(input('use verbose? (press enter for no)= '))
    initial_population_size=int(input('initial population size= '))
    n_generation=int(input('n generation= '))
    n_parents=int(input('n_parents= '))
    offspring_size=int(input('offspring size= '))
    random_population_size=int(input('random agents size= '))
    gamma=float(input('gamma= '))
    file=input('filename is ')
    start=time.time()
    with open(file,'w') as output:
        if PSO==True:
            output.write('using PSO'+"\n")
        output.write('initial population size= '+str(initial_population_size)+"\n")
        output.write('n_generation= '+str(n_generation)+"\n")
        output.write('n_parents= '+str(n_parents)+"\n")
        output.write('offspring_size= '+str(offspring_size)+"\n")
        output.write('random agents size= '+str(random_population_size)+"\n")
        output.write('gamma= '+str(gamma)+"\n")
    test=gen.GA(initial_population_size=initial_population_size,
                n_generation=n_generation,
                n_parents=n_parents,
                offspring_size=offspring_size,
                random_population_size=random_population_size,
                df_temp_food=by_20C)
    
    population,population_connection=test.running_GA(gamma=gamma,PSO=PSO,verbose=verbose,file=file)
    end=time.time()
    print(end-start)
    with open(file,'a') as output:
        output.write('total running time'+str(end-start)+"\n")
    
if __name__=='__main__':
    run_GA_func=bool(input('Run GA? (Enter to say no)'))
    if run_GA_func==True:
        run_GA()
    else:
        run_simulation()


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




