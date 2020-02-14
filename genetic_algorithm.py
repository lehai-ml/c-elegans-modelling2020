import numpy as np
import pandas as pd
import sympy as sp
from sympy.solvers import solve
from sympy import symbols, Eq
from scipy.integrate import odeint
import particleswarmop as PSO
import elegansfunc as elegans


class GA:
    def __init__(self,n_generation,population_size,n_parents,offspring_size,df_temp_food):
        self.n_generation=n_generation
        self.population_size=population_size
        self.n_parents=n_parents
        self.offspring_size=offspring_size
        self.df_temp_food=df_temp_food

    def running_GA(self):
        new_population,new_population_connections=self.random_model_generator()
        for generation in range(self.n_generation):
            print('Generation:',generation)
            mse_score=self.fitness(new_population)
            
            parents,parents_connections=self.select_mating_pool(new_population,new_population_connections,mse_score,self.n_parents)
            
            offsprings,offspring_connections=self.crossover_and_mutation(parents_connections,self.offspring_size)
            
            new_population=parents+offsprings
            new_population_connections=parents_connections+offspring_connections
            
            print('Best MSE result for this generation:',np.min(mse_score))
        best_fitness=self.fitness(new_population)
        print('Best MSE score:' np.min(best_fitness))
        print('Best solution index is:'np.argmin(best_fitness))
        return (new_population,new_population_connections)

    def random_model_generator(self):
        population=[]
        population_connections=[]
        for n in range(self.population_size):
            connections=dict({'nsm':[],'asi':[],'adf':[]})
            for i in connections:
                connections[i]=list(np.random.choice([-1,0,1],7))
            neuron_NSM=elegans.define_model_interactions('nsm',TA2TN=connections['nsm'][0],TA2DA=connections['nsm'][1],TA2S=connections['nsm'][2],TN2S=connections['nsm'][3],DA2TA=connections['nsm'][4],DA2TN=connections['nsm'][5],DA2S=connections['nsm'][6])
            neuron_ADF=elegans.define_model_interactions('adf',TA2S=connections['adf'][0],TN2TA=connections['adf'][1],TN2DA=connections['adf'][2],TN2S=connections['adf'][3],DA2TA=connections['adf'][4],DA2TN=connections['adf'][5],DA2S=connections['adf'][6])
            neuron_ASI=elegans.define_model_interactions('asi',TA2TN=connections['asi'][0],TA2DA=connections['asi'][1],TA2S=connections['asi'][2],TN2TA=connections['asi'][3],TN2DA=connections['asi'][4],TN2S=connections['asi'][5],DA2S=connections['asi'][6])
            
            model1=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
            temp_model=elegans.Model(model1,elegans.normalize_by_highest_wildtype_mean(self.df_temp_food),PSO=False)
            population.append(temp_model)
            population_connections.append(connections)
        return (population,population_connections)
    
    def fitness(self,population):
        mse_score=[model.simulate_all_together(compare=True) for model in population]
        return mse_score
    # select best of the mse_Score as parent
    
    def select_mating_pool(self, population,population_connections,mse_score, n_parents):
        parents=[]
        parents_connections=[]
        for n in range(n_parents):
            parents.append(population[np.argmin(mse_score)])
            parents_connections.append(population_connections[np.argmin(mse_score)])
            mse_score[np.argmin(mse_score)]=9999999 #so that it cannot be selected again.
        return (parents,parents_connections)
    
    def crossover_and_mutation(self,parents_connections,offspring_size):
        offsprings=[]
        offspring_connections=[]
        for k in range(offspring_size):
            parent1_idx= k%len(parents_connections)
            parent2_idx= (k+1)%len(parents_connections)
            mutation_point=np.random.choice([3,4,5],1)[0]
            mutation=list(np.random.choice([-1,0,1],1)[0])
            offspring_connection=dict({'nsm':[],'asi':[],'adf':[]})
            for i in offspring_connection:
                offspring_connection[i]=parents_connections[parent1_idx][i][0:mutation_point]+[mutation]+parents_connections[parent2_idx][i][mutation_point+1:7]
                
            neuron_NSM=elegans.define_model_interactions('nsm',TA2TN=offspring_connection['nsm'][0],TA2DA=offspring_connection['nsm'][1],TA2S=offspring_connection['nsm'][2],TN2S=offspring_connection['nsm'][3],DA2TA=offspring_connection['nsm'][4],DA2TN=offspring_connection['nsm'][5],DA2S=offspring_connection['nsm'][6])
            neuron_ADF=elegans.define_model_interactions('adf',TA2S=offspring_connection['adf'][0],TN2TA=offspring_connection['adf'][1],TN2DA=offspring_connection['adf'][2],TN2S=offspring_connection['adf'][3],DA2TA=offspring_connection['adf'][4],DA2TN=offspring_connection['adf'][5],DA2S=offspring_connection['adf'][6])
            neuron_ASI=elegans.define_model_interactions('asi',TA2TN=offspring_connection['asi'][0],TA2DA=offspring_connection['asi'][1],TA2S=offspring_connection['asi'][2],TN2TA=offspring_connection['asi'][3],TN2DA=offspring_connection['asi'][4],TN2S=offspring_connection['asi'][5],DA2S=offspring_connection['asi'][6])
            
            model2=dict({'nsm':neuron_NSM,'asi':neuron_ASI,'adf':neuron_ADF})
            temp_model=elegans.Model(model2,elegans.normalize_by_highest_wildtype_mean(self.df_temp_food),PSO=False)
            
            offsprings.append(temp_model)
            offspring_connections.append(offspring_connection)
            
        return (offsprings,offspring_connections)