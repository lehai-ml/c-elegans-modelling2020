import numpy as np
import pandas as pd
import sympy as sp
import elegansfunc as elegans

"""
Using Swarm intelligence
For each particle i:
    For each dimension d:
        Initialize position x(id) randomly within the permissible range
        Initialize vector x(id) randomly within the permissible range
While k teration<n_iteration:
    For each particle i:
        Calculate the fitness value
        If the fitness value is better than p_best(id) in ITS own history:
            update p_best(id) to current fitness value
    Choose the particle with the best fitness value as the g_best
    For each particle i:
        For each dimension d:
            Calculate velcity according to the equation:
                v(id)(k+1)= W * v(id)(k)+ c1 * rand(1)(p(id)-x(id))+ c2*rand(2)*(p(gd)-x(id))
            Update particle position according to the equation:
                x(id)(k+1)=x(id)(k)+v(id)(k+1)
    k=k+1
    If criteria reached:
        break
W = inertia weight. This affects the movement propagation given by the last velocity value
c1, c2= acceleration coef. C1 gives importance of personal best value and C2 gives social best value.
pi- best individual position 
pg- best social position.
rand1-rand2 are random numbers between 0<rand<1
"""
class Particle:
    def __init__(self,TA_present,TN_present):
        self.position=np.random.rand(2)
        self.pbest_position=self.position
        self.pbest_value=float('inf') #acts as unbounded upper value for comparision. This is useful for finding lowest values of something.
        self.velocity=np.random.rand(2)
        self.TA_present=TA_present
        self.TN_present=TN_present
        if TA_present==0:
            self.position[0]=0
            self.velocity[0]=0
        if TN_present==0:
            self.position[1]=0
            self.velocity[1]=0
    
    def __str__(self):
        print('I am at',self.position,'my p_best is',self.pbest_position)
        
    def move(self):
        self.position=self.position+self.velocity

class Space:
    def __init__(self,target, target_error,n_particles,equation_daf7,equation_WT,Snsm,Sasi,Sadf,NSM_WT,NSM_daf7,ASI_WT,ASI_daf7,ADF_WT,ADF_daf7,DA_WT,DA_daf7,TA_present,TN_present):
        self.target=target
        self.target_error=target_error
        self.n_particles=n_particles
        self.particles=[]
        self.gbest_value=float('inf')
        self.gbest_position=np.random.rand(2)
        self.equation_WT=equation_WT
        self.equation_daf7=equation_daf7
        self.W=0.5
        self.c1=0.8
        self.c2=0.9
        self.Snsm=Snsm
        self.Sasi=Sasi
        self.Sadf=Sadf
        self.NSM_WT=NSM_WT
        self.NSM_daf7=NSM_daf7
        self.ASI_WT=ASI_WT
        self.ASI_daf7=ASI_daf7
        self.ADF_WT=ADF_WT
        self.ADF_daf7=ADF_daf7
        self.DA_WT=DA_WT
        self.DA_daf7=DA_daf7
        self.TA_present=TA_present
        self.TN_present=TN_present
    
    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
    
    def fitness(self,particle):
        #def Snsm, etc.
        Snsm=self.Snsm
        Sasi=self.Sasi
        Sadf=self.Sadf
        NSM_WT=self.NSM_WT
        NSM_daf7=self.NSM_daf7
        ASI_WT=self.ASI_WT
        ASI_daf7=self.ASI_daf7
        ADF_WT=self.ADF_WT
        ADF_daf7=self.ADF_daf7
        DA_WT=self.DA_WT
        DA_daf7=self.DA_daf7
        alpha=1
        TA=particle.position[0]
        TN=particle.position[1]
        combined_equation=eval(self.equation_WT)**2+eval(self.equation_daf7)**2
        return combined_equation
    
    def set_pbest(self):
        for particle in self.particles:
            fitness_candidate=self.fitness(particle)
            
            if particle.pbest_value>fitness_candidate:#the fitness function will calculate the cell expression, you want that value to be as close to the target as possible.
                particle.pbest_value=fitness_candidate
                particle.pbest_position=particle.position
    
    def set_gbest(self):
        for particle in self.particles:
            best_fitness_candidate=self.fitness(particle)
            if(self.gbest_value>best_fitness_candidate):
                self.gbest_value=best_fitness_candidate
                self.gbest_position=particle.position
    
    def move_particles(self):
        for particle in self.particles:
            new_velocity=(self.W*particle.velocity)+(self.c1*np.random.rand())*(particle.pbest_position-particle.position)+(self.c2*np.random.rand())*(self.gbest_position-particle.position)
            if self.TA_present==0:
                new_velocity[0]=0
            if self.TN_present==0:
                new_velocity[1]=0
            particle.velocity=new_velocity
            particle.move()        