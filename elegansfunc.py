import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.solvers import solve
from sympy import symbols, Eq
from IPython.display import display, Math, Latex

def normalize_by_highest_wildtype_mean(by_15C):
    """
    We normalize the values by dividing each value to the highest fluorescent expression in the wild type neuron.
    Argument: data_frame
    """
    highest_mean_ADF=max(by_15C[by_15C['genotype']=='wildtype']['ADF'])
    highest_mean_NSM=max(by_15C[by_15C['genotype']=='wildtype']['NSM'])
    highest_mean_ASI=max(by_15C[by_15C['genotype']=='wildtype']['ASI'])
    new_value_ADF=by_15C['ADF']/highest_mean_ADF
    new_value_NSM=by_15C['NSM']/highest_mean_NSM
    new_value_ASI=by_15C['ASI']/highest_mean_ASI
    normalized_data=pd.DataFrame({'temperature':by_15C['temperature'],
                                 'Food':by_15C['Food'],
                                  'ADF':new_value_ADF,
                                  'NSM':new_value_NSM,
                                  'ASI':new_value_ASI,
                                  'tph-1':by_15C['tph-1'],
                                  'daf-7':by_15C['daf-7'],
                                  'genotype':by_15C['genotype']
                                 })
    return normalized_data

### Visualizing the previous data

# def ax_plot(ax,normalized_dataset,neuron):
"""
Plot my current datasets.
"""
#     ax.plot(normalized_dataset[normalized_dataset['genotype']=='wildtype']['Food'], 
#             normalized_dataset[normalized_dataset['genotype']=='wildtype'][neuron],'kx-',label='wildtype')
#     ax.plot(normalized_dataset[normalized_dataset['genotype']=='tph1mut']['Food'], 
#             normalized_dataset[normalized_dataset['genotype']=='tph1mut'][neuron],'bx-',label='tph1-/-')
#     ax.plot(normalized_dataset[normalized_dataset['genotype']=='daf7mut']['Food'], 
#             normalized_dataset[normalized_dataset['genotype']=='daf7mut'][neuron],'rx-',label='daf7-/-')
#     ax.plot(normalized_dataset[normalized_dataset['genotype']=='doublemut']['Food'], 
#             normalized_dataset[normalized_dataset['genotype']=='doublemut'][neuron],'mx-',label='double-/-')
#     ax.plot(np.linspace(1e0,10e10,6),np.repeat(1,6),'c--')    

# f,axes=plt.subplots(3,3,sharey=True,sharex=True,figsize=(15,10))
# neuron=['NSM','ADF','ASI']
# normalized_datasets=(normalize_by_highest_wildtype_mean(by_15C),
#                     normalize_by_highest_wildtype_mean(by_20C),
#                     normalize_by_highest_wildtype_mean(by_25C))
# temp=['15','20','25']
# for row in range(3):
#     for column in range(3):
#         ax=axes[row,column]
#         ax_plot(ax,normalized_datasets[row],neuron[column])
#         ax.set_xscale('log')
#         ax.set_xticks([1e0,1e2,1e4,1e6,1e8,1e10])
#         ax.set_xticklabels(['$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^10$'])
#         ax.set_yticks(np.linspace(0.5,1.5,11))
#         ax.tick_params(top=True,right=True,direction='in')
#         if row==0:
#             ax.set_title(neuron[column],fontdict={'fontsize':20})
#         if column==0:
#             ax.set_ylabel(temp[row]+'$^\circ$'+'C',fontsize=18)
#         if ax==axes[2,2]:
#             handles, labels = ax.get_legend_handles_labels()
            
# f.legend(handles, labels, loc='lower center',mode='expand',ncol=4,bbox_to_anchor=(0.24,0,0.5,0),
#          labelspacing=1.2,columnspacing=5,fontsize='x-large',frameon=False)
# f.text(0.5, 0.06, 'Food Levels (cells/ml)', ha='center',size='xx-large')
# f.text(0.05, 0.5, 'Normalised Fluorescent Expression AU', va='center', rotation='vertical',size='xx-large')
# f.suptitle('Normalized Fluorescent Expression by Food Level across temperature and neurons',size='xx-large')
# plt.show()

def define_model_interactions(neuron_label,
                             TA2TA=0,TA2TN=0,TA2DA=0,TA2S=0,
                             TN2TA=0,TN2TN=0,TN2DA=0,TN2S=0,
                             DA2TA=0,DA2TN=0,DA2DA=0,DA2S=0):
    """
    Define neuron interactions: argument defines direction of interaction (TO THE CELL and to connections to the cell).
    
    TA- tph-1 activity from ADF. TN- tph-1 activity from NSM. DA- daf-7
    activity from ASI. S defines the cell body. (all self-regulatory action)
    You define a table, where the direction is row->column.
    The values are -1 - inhibitory, 0- no activity, and 1- promotory.
    
    Constraints:Valid only for max 3 neurons.
    Each neuron can receive max 3 DIRECT connections (i.e. 2 from 2 unexamined cells and 1 self-regulatory).
    The self-regulatory can receive up to 2 further connections from the other two cells.
    The other 2 DIRECT connections, can receive up to 1 connection each from the other unexamined cell.
    These connections will be referred to as INDIRECT connections.
    Example: NSM is examined cell. It can receive TN2S, TA2S, DA2S as DIRECT CONNECTIONS.
    It can receive TA2TN,DA2TN, DA2TA and TA2DA as INDIRECT CONNECTIONS. To clarify, these indirect connections are from 1 unexamined cell to DIRECT connection of another unexamined cells (i.e. DA2TA is the connection from ASI to the TA2S connection, not to TA2TN connection)
    
    """
    neuron=pd.DataFrame(({'TA':np.zeros(3,dtype='int'),
                  'TN':np.zeros(3,dtype='int'),
                  'DA':np.zeros(3,dtype='int'),
                  'S':np.zeros(3,dtype='int')}),index=['TA','TN','DA'])
    
    if (neuron_label=='nsm'):
        TN2TA=TN2DA=0
    elif (neuron_label=='asi'):
        DA2TN=DA2TA=0
    elif (neuron_label=='adf'):
        TA2TN=TA2DA=0
            
    neuron.loc['TA','TA']=0
    neuron.loc['TA','TN']=TA2TN
    neuron.loc['TA','DA']=TA2DA
    neuron.loc['TA','S']=TA2S
    neuron.loc['TN','TA']=TN2TA
    neuron.loc['TN','TN']=0
    neuron.loc['TN','DA']=TN2DA
    neuron.loc['TN','S']=TN2S
    neuron.loc['DA','TA']=DA2TA
    neuron.loc['DA','TN']=DA2TN
    neuron.loc['DA','DA']=0
    neuron.loc['DA','S']=DA2S
    
    neuron.columns=neuron.columns+neuron_label
    neuron.index=neuron.index+neuron_label
    
    return neuron

class Neuron:
    """
    Define a neuron, that has a defined model interaction.
    """
    def __init__(self,neuron,neuron_label,df_temp_food):
        self.neuron=neuron
        self.neuron_label=neuron_label
        self.df_temp_food=df_temp_food
        
        
    def write_equation(self,print_equation=False):
        """
        Write the differential equation in string format.
        Please read the define_model_interaction_function above.
        Function_work_flow: First, define the DIRECT activatory connections to the cell. (list1)
        Then define the INDIRECT activatory connections to those in list1.(list2)
        Then define the INDIRECT inhibitory connections to those in list1. (dict1)
        Then define the DIRECT inhibitory connections to the cells. (list3)
        Then define the INDIRECT activatory connections to those in list3. (list4)
        Then define the INDIRECT inhibitory connections to those in list2 (dict2)
        
        Then build term by term:
        S term=S/denominator for S term (saved as string), where denominator for S term = everything in dict2(already saved as equation)+everything in list4.
        Direct activation term= take everything from dict1 (already saved as equation)+everything in list2.
        
        Equation= join S term with Direct activation term with degradation term.
        print_equation=True-> print as Latex, works in IPython.
        
        """
        neuron=self.neuron
        neuron_temp=neuron.copy()
        neuron_label=self.neuron_label

        neuron_temp.columns=neuron_temp.columns+('*ADF','*NSM','*ASI','')
        neuron_temp.index=neuron_temp.index+('*ADF','*NSM','*ASI')

        interactions_name=list(neuron_temp.index)


        ###activatory_interactions
        direct_activatory_interactions_to_cell=[interactions_name[i] for i in np.where(neuron_temp['S{}'.format(neuron_label)]==1)[0]] #anything that directly activate the cell

        indirect_activation_to_activatory_interactions_to_cell=[interactions_name[i] for n in direct_activatory_interactions_to_cell for i in np.where(neuron_temp[n]==1)[0]]#anything that activate the direct activation of the cell.

        indirect_inhibition_to_activatory_interactions_to_cell=dict()
        for n in direct_activatory_interactions_to_cell:
            indirect_inhibition_to_activatory_interactions_to_cell[n]=[interactions_name[i] for i in np.where(neuron_temp[n]==-1)[0]]
        for i in indirect_inhibition_to_activatory_interactions_to_cell:
            if len(indirect_inhibition_to_activatory_interactions_to_cell[i])==0:
                indirect_inhibition_to_activatory_interactions_to_cell[i]='1'
            else:
                indirect_inhibition_to_activatory_interactions_to_cell[i]='('+'+'.join(['1']+indirect_inhibition_to_activatory_interactions_to_cell[i])+')'


        ###inhibitory_interactions these will be all under the S
        direct_inhibitory_interactions_to_cell=[interactions_name[i] for i in np.where(neuron_temp['S{}'.format(neuron_label)]==-1)[0]]

        indirect_activation_to_inhibitory_interactions_to_cell=[interactions_name[i] for n in direct_inhibitory_interactions_to_cell for i in np.where(neuron_temp[n]==1)[0]]

        indirect_inhibition_to_inhibitory_interactions_to_cell=dict()
        for n in direct_inhibitory_interactions_to_cell:
            indirect_inhibition_to_inhibitory_interactions_to_cell[n]=[interactions_name[i] for i in np.where(neuron_temp[n]==-1)[0]]
        for i in indirect_inhibition_to_inhibitory_interactions_to_cell:
            if len(indirect_inhibition_to_inhibitory_interactions_to_cell[i])==0:
                indirect_inhibition_to_inhibitory_interactions_to_cell[i]='1'
            else:
                indirect_inhibition_to_inhibitory_interactions_to_cell[i]='('+'+'.join(['1']+indirect_inhibition_to_inhibitory_interactions_to_cell[i])+')'


        ###build term by term:
        #term Snsm:

        denominator_of_S=[]
        for i in list(indirect_inhibition_to_inhibitory_interactions_to_cell.keys()):
            term='/'.join([i,list(indirect_inhibition_to_inhibitory_interactions_to_cell.values())[0]])
            denominator_of_S.append(term)
        for i in indirect_activation_to_inhibitory_interactions_to_cell:
            denominator_of_S.append(i)
        if denominator_of_S==[]:
            denominator_of_S=['1']
        else:
            denominator_of_S=['('+'+'.join(['1']+denominator_of_S)+')']
        S_term=['/'.join(['S{}'.format(neuron_label)]+denominator_of_S)]

        ##term for direct activation of S:
        direct_activation_term=[]
        for i in list(indirect_inhibition_to_activatory_interactions_to_cell.keys()):
            term='/'.join([i,list(indirect_inhibition_to_activatory_interactions_to_cell.values())[0]])
            direct_activation_term.append(term)
        for i in indirect_activation_to_activatory_interactions_to_cell:
            direct_activation_term.append(i)
        direct_activation_term=['+'.join(direct_activation_term)]


        equation='+'.join(S_term+direct_activation_term)+'-alpha*{}'.format(neuron_label.upper())

        if print_equation:        
            return display(Math('dX{}/dt='.format(neuron_label)+equation))
        else:
            return equation
    
    def find_DA_at_food_level(self,food,alpha=1): 
        """
        Find DA values at defined food level from double mutants and tph-1 mutants.
        Evaluate from previous write_equation function necessary symbols, and solve using sympy.
        alpha=1 default.
        
        Important: If there are no DA defined, or no solution for DA is found, solution=0. Shouldn't matter because denominator will never be 0.
        """
        
        neuron=self.neuron
        neuron_label=self.neuron_label
        df_temp_food=self.df_temp_food
        df_temp_food=df_temp_food[df_temp_food['Food']==food]
        
        TAnsm=TAadf=TAasi=0
        TNnsm=TNadf=TNasi=0
        
        
        Snsm=np.array(df_temp_food['NSM'][df_temp_food['genotype']=='doublemut'])
        Sasi=np.array(df_temp_food['ASI'][df_temp_food['genotype']=='doublemut'])
        Sadf=np.array(df_temp_food['ADF'][df_temp_food['genotype']=='doublemut'])
        
        NSM=np.array(df_temp_food['NSM'][df_temp_food['genotype']=='tph1mut'])
        ASI=np.array(df_temp_food['ASI'][df_temp_food['genotype']=='tph1mut'])
        ADF=np.array(df_temp_food['ADF'][df_temp_food['genotype']=='tph1mut'])
        
        X=symbols('X')
        equation_to_solve=self.write_equation().replace('DA'+neuron_label,'X')
        solution=solve(eval(equation_to_solve),X,dict=True)
        if solution==[]:
            solution=0
            return solution
        else:
            return solution[0][X]


    def find_TA_TN_in_daf7mut(self,food,alpha=1):
        """
        in WT and daf7 mut
        
        After solving for DA, we can solve for TA and TN. We rewrite them as set of linear equations. For daf-7 mutant, the DA=0.
        we extract the TA+neuron label. and TN+neuron label depending on the neuron we looking at.
        we need only simplify equation to the TA=TN+something, next step we put TN+something-TA=0 as our linear eq.
        """
        neuron=self.neuron
        neuron_label=self.neuron_label
        df_temp_food=self.df_temp_food
        df_temp_food=df_temp_food[df_temp_food['Food']==food]
        
        Snsm,Sasi,Sadf,NSM,ASI,ADF=symbols('Snsm Sasi Sadf NSM ASI ADF')
        DAnsm,DAadf,DAasi=symbols('DAnsm DAadf DAasi')
        TAnsm,TAadf,TAasi=symbols('TAnsm TAadf TAasi')
        TNnsm,TNadf,TNasi=symbols('TNnsm TNadf TNasi')
        alpha=symbols('alpha')
        
        X1=symbols('TA'+neuron_label)
        X2=symbols('TN'+neuron_label)
        
        
        equation_to_solve=self.write_equation()
        equation=Eq(eval(equation_to_solve))
        TA_TN_daf7mut=sp.solve(equation.subs({Snsm:np.array(df_temp_food['NSM'][df_temp_food['genotype']=='doublemut']),
                                              Sasi:np.array(df_temp_food['ASI'][df_temp_food['genotype']=='doublemut']),
                                              Sadf:np.array(df_temp_food['ADF'][df_temp_food['genotype']=='doublemut']),
                                              NSM:np.array(df_temp_food['NSM'][df_temp_food['genotype']=='daf7mut']),
                                              ASI:np.array(df_temp_food['ASI'][df_temp_food['genotype']=='daf7mut']),
                                              ADF:np.array(df_temp_food['ADF'][df_temp_food['genotype']=='daf7mut']),
                                              DAnsm:0,DAadf:0,DAasi:0,alpha:1
                                             }),(X1,X2))
        return TA_TN_daf7mut
        
        
    def find_TA_TN_in_WT(self,food,alpha=1):
        """
        **see help(find_TA_TN_in_daf7mut)**
        """
        neuron=self.neuron
        neuron_label=self.neuron_label
        df_temp_food=self.df_temp_food
        df_temp_food=df_temp_food[df_temp_food['Food']==food]
        
        Snsm,Sasi,Sadf,NSM,ASI,ADF=symbols('Snsm Sasi Sadf NSM ASI ADF')
        DAnsm,DAadf,DAasi=symbols('DAnsm DAadf DAasi')
        TAnsm,TAadf,TAasi=symbols('TAnsm TAadf TAasi')
        TNnsm,TNadf,TNasi=symbols('TNnsm TNadf TNasi')
        alpha=symbols('alpha')
        
        X1=symbols('TA'+neuron_label)
        X2=symbols('TN'+neuron_label)
        
        equation_to_solve=self.write_equation()
        equation=Eq(eval(equation_to_solve))
        
        TA_TN_WT=sp.solve(equation.subs({Snsm:np.array(df_temp_food['NSM'][df_temp_food['genotype']=='doublemut']),
                                                 Sasi:np.array(df_temp_food['ASI'][df_temp_food['genotype']=='doublemut']),
                                                 Sadf:np.array(df_temp_food['ADF'][df_temp_food['genotype']=='doublemut']),
                                                 NSM:np.array(df_temp_food['NSM'][df_temp_food['genotype']=='wildtype']),
                                                 ASI:np.array(df_temp_food['ASI'][df_temp_food['genotype']=='wildtype']),
                                                 ADF:np.array(df_temp_food['ADF'][df_temp_food['genotype']=='wildtype']),
                                                 DAnsm:self.find_DA_at_food_level(food),
                                                 DAadf:self.find_DA_at_food_level(food),
                                                 DAasi:self.find_DA_at_food_level(food),
                                                 alpha:1
                                             }),(X1,X2))
        
        return TA_TN_WT
    
    
    
    def systems_of_equations(self,food,alpha=1):
        """
        Use the TA_TN_WT/daf_7mut equations and solve them as systems of equations (could be linear or non-linear).
        **see help(find_TA_TN_in_daf7mut)**

        Important: In case there is no solution to the system:
        i.e. eq1=TAnsm=9, but eq2=TAnsm=3, both answers are saved.
        
        """
        neuron=self.neuron
        neuron_label=self.neuron_label
        df_temp_food=self.df_temp_food
        df_temp_food=df_temp_food[df_temp_food['Food']==food]
        
        TA_TN_daf7mut=self.find_TA_TN_in_daf7mut(food,alpha=1)
        TA_TN_WT=self.find_TA_TN_in_WT(food,alpha=1)
        
        X1=symbols('TA'+neuron_label)
        X2=symbols('TN'+neuron_label)
        
        if (TA_TN_daf7mut==False or TA_TN_WT==False):
            solutions=dict({'TA'+neuron_label:0,
                            'TN'+neuron_label:0})
            ##In case there is no solution due to no TA or TN, set those value to 0
        else:
            daf_7_equation=[Eq(TA_TN_daf7mut[0][i]-i) for i in TA_TN_daf7mut[0]][0]
            WT_equation=[Eq(TA_TN_WT[0][i]-i) for i in TA_TN_WT[0]][0]
        
            solutions=sp.solve((daf_7_equation,WT_equation))
            ##in case where there is no solution to the system of equation, return all the examined values
            if len(solutions)==0:
                solutions=[sp.solve(daf_7_equation,(X1,X2))[0],sp.solve(WT_equation,(X1,X2))[0]]
                if solutions[0].keys()==solutions[1].keys():
                    #if the they are the same keys, just take the average
                    key=list(solutions[0].keys())[0]
                    value1=list(solutions[0].values())[0]
                    value2=list(solutions[1].values())[0]
                    solutions=dict({key:np.mean([value1,value2])})
        if (type(solutions)==list and len(solutions)>1):
            ###If there are more than one set of solutions, choose the one with higher mean between the TA and TN.
            hi_mean=np.argmax([np.mean(list(i.values())) for i in solutions])
            solutions=solutions[hi_mean]##this will ensure all output will be dict    
        return solutions
    
class Model:
    """
    save all defined neurons together (max 3).
    Argument: object that has defined_model_interaction.
    """
    def __init__(self,new_model,df_temp_food):
        #input a dict with defined_model_interaction neuron
        self.model=dict()
        for i in new_model:
            self.model[i]=Neuron(new_model[i],i,df_temp_food)            
        self.df_temp_food=df_temp_food
    
    # def model_interactions(self): 
    #     #display model interactions in form of table side by side
    #     html_str=''
    #     for i in self.model:
    #         df=self.model[i].neuron
    #         html_str+=df.to_html()
    #     display_html(html_str.replace('table','table style="display:inline"'),raw=True)

    def DA_table_across_food_level(self,round_number=False,number=3): 
        #call Neuron.find DA_at_food_level for all neurons in the model dict.
        table=pd.DataFrame({'Food':np.unique(self.df_temp_food['Food'])})
        for i in self.model:
            if round_number==True:
                table['DA'+i]=[round(self.model[i].find_DA_at_food_level(n),number) for n in np.unique(self.df_temp_food['Food'])]
            else:
                table['DA'+i]=[self.model[i].find_DA_at_food_level(n) for n in np.unique(self.df_temp_food['Food'])]
        return table
    
    def TA_TN_table_across_food_levels(self):
        """
        Calculate the TA_TN across 6 food levels, and append them into a dataframe.
        """
        solutions=[]
        for i in self.model:
            solutions.append([self.model[i].systems_of_equations(n) for n in np.unique(self.df_temp_food['Food'])])
        TA_TN_pd=pd.DataFrame({'Food':np.unique(self.df_temp_food['Food']),
                               'TAnsm':np.zeros(6),'TAadf':np.zeros(6),'TAasi':np.zeros(6),'TNnsm':np.zeros(6),'TNadf':np.zeros(6),'TNasi':np.zeros(6)})
        for i in solutions:
            for n in range(6):
                for x in list(i[n].keys()):
                    TA_TN_pd.loc[n,str(x)]=i[n][x]
      
        return TA_TN_pd