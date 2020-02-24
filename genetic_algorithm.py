import numpy as np
import pandas as pd
import sympy as sp
from sympy.solvers import solve
from sympy import symbols, Eq
from scipy.integrate import odeint
import particleswarmop as PSO
import elegansfunc as elegans
"""
Initialize with n random models
For each generation in defined number generations:
    Calculate the fitness value (i.e. MSE score)
    Choose p number of parents to create o number of offsprings.
    Each offsprings are further mutated so that the population at the end one generation is consisted of mutated offsprings and their parents.
return the last population along with the index of the best model.
"""

class GA:
    def __init__(self,initial_population_size,n_generation,n_parents,offspring_size,random_population_size,dataset_dict):
        self.initial_population_size=initial_population_size
        self.random_population_size=random_population_size
        self.n_generation=n_generation
        self.n_parents=n_parents
        self.offspring_size=offspring_size
        self.dataset_dict=dataset_dict
            

    def running_GA(self,gamma=0.5,PSO=False,verbose=False,file='generation.txt'):
        output=open(file,'a')
        new_population,new_population_connections=self.random_model_generator(population_size=self.initial_population_size,PSO=PSO,verbose=verbose)
        output.close()
        for generation in range(self.n_generation):
            output=open(file,'a')
            print('Generation: ',generation)
            output.write('Generation '+str(generation)+"\n")
            [output.write(str(i)+"\n") for i in new_population_connections]
            mse_score,temp_score_dict=self.fitness(new_population,gamma)
            print('Detailed mse by temperature for each model',temp_score_dict)
            print(mse_score)
            print('Average MSE result for this generation: ',np.mean(mse_score))
            print('Best MSE result for this generation: ',np.min(mse_score))
            
            output.write('Detailed mse by temperature for each model'+ str(temp_score_dict)+"\n")
            output.write('MSE score '+str(mse_score)+"\n")
            output.write('Average MSE result for this generation '+str(np.mean(mse_score))+"\n")
            output.write('Best MSE result for this generation is '+str(np.min(mse_score))+"\n")
            
            parents,parents_connections=self.select_mating_pool(new_population,new_population_connections,mse_score)
            
            offsprings,offspring_connections=self.crossover_and_mutation(parents_connections,PSO=PSO,verbose=verbose)
            
            random_agents,random_agents_connections=self.random_model_generator(population_size=self.random_population_size,PSO=PSO,verbose=verbose)
            
            
            new_population=parents+offsprings+random_agents
            new_population_connections=parents_connections+offspring_connections+random_agents_connections
            
            output.close()
        output=open(file,'a')
        print('Final population')
        output.write('Final population'+"\n")
        [output.write(str(i)+"\n") for i in new_population_connections]
        best_fitness,temp_score_dict=self.fitness(new_population,gamma)
        output.write('Detailed mse by temperature for each model'+ str(temp_score_dict)+"\n")
        output.write('MSE score '+str(best_fitness)+"\n")
        print('Best MSE score: ',np.min(best_fitness))
        output.write('Best MSE score is '+str(np.min(best_fitness))+"\n")
        output.write('Average MSE result for this generation: '+str(np.mean(best_fitness))+"\n")
        print('Best solution index is:',np.argmin(best_fitness))
        output.write('Best solution index is '+str(np.argmin(best_fitness))+"\n")
        output.close()
        return (new_population,new_population_connections)

    def random_model_generator(self,population_size,PSO,verbose):
        population=[]
        population_connections=[]
        for n in range(population_size):
            print('creating random model ',n+1)
            connections=dict({'nsm':[],'asi':[],'adf':[]})
            for i in connections:
                connections[i]=list(np.random.choice([-1,0,1],7))
            neuron_NSM=elegans.define_model_interactions('nsm',TA2TN=connections['nsm'][0],TA2DA=connections['nsm'][1],TA2S=connections['nsm'][2],TN2S=connections['nsm'][3],DA2TA=connections['nsm'][4],DA2TN=connections['nsm'][5],DA2S=connections['nsm'][6])
            neuron_ADF=elegans.define_model_interactions('adf',TA2S=connections['adf'][0],TN2TA=connections['adf'][1],TN2DA=connections['adf'][2],TN2S=connections['adf'][3],DA2TA=connections['adf'][4],DA2TN=connections['adf'][5],DA2S=connections['adf'][6])
            neuron_ASI=elegans.define_model_interactions('asi',TA2TN=connections['asi'][0],TA2DA=connections['asi'][1],TA2S=connections['asi'][2],TN2TA=connections['asi'][3],TN2DA=connections['asi'][4],TN2S=connections['asi'][5],DA2S=connections['asi'][6])
            
            model1=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
            print(model1)
            
            temp_model=dict()
            for i in self.dataset_dict:
                temp_model[i]=elegans.Model(model1,elegans.normalize_by_highest_wildtype_mean(self.dataset_dict[i]),PSO=PSO,verbose=verbose)
                                
            population.append(temp_model)
            population_connections.append(connections)
        return (population,population_connections)
    
    def fitness(self,population,gamma):
        mse_score=[]
        temp_score_dict=dict()
        for idx,model_dict in enumerate(population):
            temp_score=[model_dict[i].simulate_all_together(gamma=gamma,compare=True) for i in model_dict]
            temp_score_dict[idx]=temp_score
            mse_score.append(np.mean(temp_score))
        return (mse_score,temp_score_dict)
    # select best of the mse_Score as parent
    
    def select_mating_pool(self, population,population_connections,mse_score):
        print('now selecting parents for mating')
        parents=[]
        parents_connections=[]
        for n in range(self.n_parents):
            parents.append(population[np.argmin(mse_score)])
            parents_connections.append(population_connections[np.argmin(mse_score)])
            mse_score[np.argmin(mse_score)]=9999999 #so that it cannot be selected again.
        return (parents,parents_connections)
    
    def crossover_and_mutation(self,parents_connections,PSO,verbose):
        print('now creating offsprings')
        offsprings=[]
        offspring_connections=[]
        for k in range(self.offspring_size):
            parent1_idx= k%len(parents_connections)
            parent2_idx= (k+1)%len(parents_connections)
            mutation_point=np.random.choice([3,4,5],1)[0]
            mutation=np.random.choice([-1,0,1],1)[0]
            offspring_connection=dict({'nsm':[],'asi':[],'adf':[]})
            for i in offspring_connection:
                offspring_connection[i]=parents_connections[parent1_idx][i][0:mutation_point]+[mutation]+parents_connections[parent2_idx][i][mutation_point+1:7]
                
            neuron_NSM=elegans.define_model_interactions('nsm',TA2TN=offspring_connection['nsm'][0],TA2DA=offspring_connection['nsm'][1],TA2S=offspring_connection['nsm'][2],TN2S=offspring_connection['nsm'][3],DA2TA=offspring_connection['nsm'][4],DA2TN=offspring_connection['nsm'][5],DA2S=offspring_connection['nsm'][6])
            neuron_ADF=elegans.define_model_interactions('adf',TA2S=offspring_connection['adf'][0],TN2TA=offspring_connection['adf'][1],TN2DA=offspring_connection['adf'][2],TN2S=offspring_connection['adf'][3],DA2TA=offspring_connection['adf'][4],DA2TN=offspring_connection['adf'][5],DA2S=offspring_connection['adf'][6])
            neuron_ASI=elegans.define_model_interactions('asi',TA2TN=offspring_connection['asi'][0],TA2DA=offspring_connection['asi'][1],TA2S=offspring_connection['asi'][2],TN2TA=offspring_connection['asi'][3],TN2DA=offspring_connection['asi'][4],TN2S=offspring_connection['asi'][5],DA2S=offspring_connection['asi'][6])
            
            model2=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
            print(model2)
            
            temp_model=dict()
            for i in self.dataset_dict:
                temp_model[i]=elegans.Model(model2,elegans.normalize_by_highest_wildtype_mean(self.dataset_dict[i]),PSO=PSO,verbose=verbose)
            
            offsprings.append(temp_model)
            offspring_connections.append(offspring_connection)
            
        return (offsprings,offspring_connections)