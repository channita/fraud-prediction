# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:51:47 2016

@author: Channita
"""

print ("IMPORTING LIBRARIES...")

import pandas as pd
import numpy as np
import re
import requests
from requests.auth import HTTPBasicAuth
from deap import creator, base, tools, algorithms #GENETIC ALGORITHM LIBRARY - requirement: pip install deap
from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import Imputer
from sklearn import metrics

#DOWLOADING FILE FROM DROPBOX FIRST TIME

import urllib.request as urllib2
import os.path
import time
import random
while not os.path.exists('dev.csv') or not os.path.exists('oot0.csv'):

    time.sleep (3*random.random()); #Sleeping less than 3 seconds before going to Dropbox - avoid too many students at once.

    if not os.path.exists('dev.csv'):
        print ("DOWLOADING FILE dev.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://dl.dropboxusercontent.com/u/28535341/dev.csv")
        output = open('dev.csv','wb')
        output.write(csvfile.read())
        output.close()

    if not os.path.exists('oot0.csv'):
        print ("DOWLOADING FILE oot0.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://dl.dropboxusercontent.com/u/28535341/oot0.csv")
        output = open('oot0.csv','wb')
        output.write(csvfile.read())
        output.close()  

#DOWLOADING FILE FROM DROPBOX FIRST TIME
    
#%%
    
print ("LOADING DATASETS...")

df = pd.read_csv("dev.csv") #DEV-SAMPLE
dfo = pd.read_csv("oot0.csv") #OUT-OF-TIME SAMPLE

#%%

print ("IDENTIFYING TYPES...")

in_model = []
list_ib = []  #input binary
list_icn = [] #input categorical nominal
list_ico = [] #input categorical ordinal
list_if = []  #input numerical continuos (input float)
list_inputs = []

output_var = 'ob_target'

for var_name in df.columns:

    if re.search('^i',var_name):
        list_inputs.append(var_name)
        print (var_name,"is input")

    if re.search('^ib_',var_name):
        list_ib.append(var_name)
        print (var_name,"is input binary")

    elif re.search('^icn_',var_name):
        list_icn.append(var_name)
        print (var_name,"is input categorical nominal")

    elif re.search('^ico_',var_name):
        list_ico.append(var_name)
        print (var_name,"is input categorical ordinal")

    elif re.search('^if_',var_name):
        list_if.append(var_name)
        print (var_name,"is input numerical continuos (input float)")

    elif re.search('^ob_',var_name):
        output_var = var_name

    else:
        print ("ERROR: unable to identify the type of:", var_name)

#%%

print ("GENETIC ALGORITHM FOR FEATURE SELECTION:")

#####
#SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
#####

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(list_inputs))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

NPOPSIZE = 50 #RANDOM STARTING POOL SIZE
population = toolbox.population(n=NPOPSIZE)

#%%

#####
#ASSESSING GINI ON THE STARTING POOL
#####
dic_gini={}
for i in range(np.shape(population)[0]): 
    # TRASLATING DNA INTO LIST OF VARIABLES (1-81)
    var_model = []    
    for j in range(np.shape(population)[0]): 
        if (population[i])[j]==1:
            var_model.append(list(list_inputs)[j])

# ASSESSING GINI INDEX FOR EACH INVIVIDUAL IN THE INITIAL POOL 
X_train=df[var_model]
Y_train=df[output_var]

rf = RandomForestRegressor(n_estimators = 400, max_features = 'log2', oob_score = True, random_state = 0, warm_start = True)
result = rf.fit(X_train, Y_train)
print(result.score(X_train, Y_train))
Y_predict = result.predict(X_train)

# HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
auc = metrics.auc(fpr, tpr)
gini_score = 2*auc-1

######
# CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
#####                
    
gini=str(gini_score)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
dic_gini[gini]=population[j]   
list_gini=sorted(dic_gini.keys(),reverse=True)

#%%#####
#GENETIC ALGORITHM MAIN LOOP - START
# - ITERATING MANY TIMES UNTIL NO IMPROVMENT HAPPENS IN ORDER TO FIND THE OPTIMAL SET OF CHARACTERISTICS (VARIABLES)
#####
sum_current_gini=0.0
sum_current_gini_1=0.0
sum_current_gini_2=0.0
first=0    
OK = 1
a=0
while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
    a=a+1
    print ('loop '), a
    OK=0

    ####
    # GENERATING OFFSPRING - START
    ####
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population =toolbox.select(offspring, k=len(population))
    ####
    # GENERATING OFFSPRING - END
    ####

sum_current_gini_2=sum_current_gini_1
sum_current_gini_1=sum_current_gini
sum_current_gini=0.0

    #####
    #ASSESSING GINI ON THE OFFSPRING - START
    #####
for j in range(np.shape(population)[0]): 
    if population[j] not in dic_gini.values(): 
       var_model = [] 
       for i in range(np.shape(population)[0]): 
           if (population[j])[i]==1:
                var_model.append(list(list_inputs)[i])
            
X_train=df[var_model]
Y_train=df[output_var]
            
            ######
            # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
            #####            
rf = RandomForestRegressor(n_estimators = 400, max_features = 'log2', oob_score = True, random_state = 0, warm_start = True)
result = rf.fit(X_train, Y_train)
print(result.score(X_train, Y_train))
Y_predict = result.predict(X_train)
            ######
            # CHANGE_HERE - END: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
            #####            
 
            
            ######
            # CHANGE_HERE - START: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
            #####                       
fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
auc = metrics.auc(fpr, tpr)
gini_score = 2*auc-1
            ######
            # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
            #####                       
           
gini=str(gini_score)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
dic_gini[gini]=population[j]  
    #####
    #ASSESSING GINI ON THE OFFSPRING - END
    #####

    #####
    #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
    #####           
list_gini=sorted(dic_gini.keys(),reverse=True)
population=[]
for i in list_gini[:NPOPSIZE]:
    population.append(dic_gini[i])
    gini=float(i.split(';')[0])
    sum_current_gini+=gini
    #####
    #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - END
    #####           
      
    #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
print ('sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2)
if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
    OK=1
    
#####
#GENETIC ALGORITHM MAIN LOOP - END
#####
        
#%%
        
gini_max=list_gini[0]        
gini=float(gini_max.split(';')[0])
features=gini_max.split(';')[1]


####
# PRINTING OUT THE LIST OF FEATURES
#####
f=0
for i in range(len(features)):
    if features[i]=='1':
        f+=1
        print ('feature ', f, ':', list(list_inputs)[i])
print ('gini: ', gini)

#%%




#%%

print ("STEP 4: ASSESSING THE MODEL...")

print ("GINI DEVELOPMENT=", gini)

print ("STEP 5: SUBMITTING THE RESULTS...")

dfo['pred'] = Y_predict
a = ['id','pred']
dfo_tosend = dfo[a]

i=1
filename = "group_Z_sub"+str(i)+".csv"
dfo_tosend.to_csv(filename, sep=',')

url = 'http://mgadi.pythonanywhere.com/api/v1.0/uploadpredictions'

files = {'file': (filename, open(filename, 'rb'))}
rsub = requests.post(url, files=files, auth=HTTPBasicAuth('user20', 'channguyen87'))
resp_str = str(rsub.text)

print ("RESULT SUBMISSION: ", resp_str)
