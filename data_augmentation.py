


# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:20:34 2020

@author: ellen.oosting
"""


%reset -f

# laad libraries
from pyteomics import mzml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

# waar staan de files
pad="C://Users//ellen.oosting//Documents//MALDI//tweedezending//MALDI//mzML//big_output//"

#filename=pad

import os
os.chdir("C://Users//ellen.oosting//Documents//MALDI//tweedezending//MALDI")
from read_final import *

metaData,maldi=importBruker(pad)


# lees classificatie data in
# en voeg metadata er aan toe
fileNameClassification="C://Users//ellen.oosting//Documents//MALDI//tweedezending//MALDI//MaldiClassification.csv"
classificationData = pd.read_csv(fileNameClassification,sep=";")
classificationData.rename(columns={"Id":"BrukerId"},inplace=True)

metaData=metaData.merge(classificationData,on=["BrukerId","Position","Identification"])
Maldidata=metaData[(metaData["Id"].notnull()) & (metaData["FirstMatch"] != "not reliable indentification")]
maldiData=Maldidata.merge(maldi,on=["Id"])

del(maldi,fileNameClassification,pad,Maldidata)


random.seed(123)
#random.sample(range(1,10),4)
#[[1, 5, 9, 4]
    
#############################################################################        


from supporting_functions import *





dataset=maldiData.copy()
wantednumbersofrecords=300
horizontal_values=[2,25]
distribution_augmentation={"horizontal":[40],
                               "add_noise":[20],
                               "linear_line":[20],
                               "multiplication":[20],
                               "offset":[20]}


# delen door 100



def random_augmentation(dataset,wantednumbersofrecords=300,horizontal_values=[2,25],distribution_augmentation={"horizontal":[40],
                               "add_noise":[20],
                               "linear_line":[20],
                               "multiplication":[20],
                               "offset":[20]}):
    # distributie kans op welke augmentation    
    
    # maak alvast result
    result=dataset.copy()
    
    # meta data bijhouden
    meta=pd.DataFrame.from_dict({"old_ID":[],"new_ID":[],"augmentation":[],"done":[],"time":[]})
    
    nieuweID=dataset["Id"].max()
    
    
    
    while result.shape[0]<(wantednumbersofrecords):
        
        #print(result.shape[0])
        #Id = random.sample(range(min(dataset["Id"]), max(dataset["Id"])), 1)[0]
        
        # welk record pakken we om wat mee te doen?
        Id=random.randint(min(dataset["Id"]), max(dataset["Id"]))

        # welke augmentation
        # gooi met de dobbelsteen om te bepalen wat voor augmentation er gaat gebeuren
        dobbelsteen=random.randint(1,6)
        # maak result dataframe
        
        data=dataset.values.copy()
        
        
        if dobbelsteen==1:
            if random.randint(0,101)<=distribution_augmentation["horizontal"][0]:
                # doe augmentation
                # horizontal shift

                tussenres=horizontal(dataset,horizontal_values[0],horizontal_values[1],Id)
                tussenres["Id"]=result["Id"].max()+1
                result=result.append(tussenres)
                result=result.reset_index(drop=True)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(result["Id"].max())],"augmentation":["horizontal"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["horizontal"],"done":["no"],"time":[datetime.now()]})])       
               
        elif dobbelsteen ==2:
            if random.randint(0,101)<=distribution_augmentation["add_noise"][0]:
                # add noise
                tussenres=add_noise(dataset,Id)
                tussenres["Id"]=result["Id"].max()+1
                result=result.append(tussenres)
                result=result.reset_index(drop=True)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(result["Id"].max())],"augmentation":["add_noise"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["add_noise"],"done":["no"],"time":[datetime.now()]})])
            
            
        elif dobbelsteen ==3:
            if random.randint(0,101)<=distribution_augmentation["linear_line"][0]:
                # linear line subtraction
                tussenres=linear_line(dataset,Id)
                tussenres["Id"]=result["Id"].max()+1
                result=result.append(tussenres)
                result=result.reset_index(drop=True)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(result["Id"].max())],"augmentation":["linear_line"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["linear_line"],"done":["no"],"time":[datetime.now()]})])
                        
        elif dobbelsteen == 4:
            if random.randint(0,101)<=distribution_augmentation["multiplication"][0]:
                tussenres=pd.DataFrame(data=np.concatenate([dataset.iloc[Id,0:9].values,dataset.iloc[Id,9:].values*random.uniform(0.90,1.10)], axis=0)).T
                tussenres.columns=dataset.columns
                tussenres["Id"]=nieuweID
                result=result.append(tussenres)
                result=result.reset_index(drop=True)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["multiplication"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["multiplication"],"done":["no"],"time":[datetime.now()]})])
            
        elif dobbelsteen ==5:
            if random.randint(0,101)<=distribution_augmentation["offset"][0]:
                
                randomvalue=random.uniform(-0.10,0.10)*dataset.iloc[Id,9:].values.std()
                tussenres=dataset.iloc[Id,9:].apply(lambda x: x+randomvalue)
                tussenres[tussenres<0]=0
                tussenres=pd.DataFrame(np.concatenate((data[Id,0:9],tussenres))).T   
                tussenres.columns=dataset.columns
                tussenres["Id"]=nieuweID  # gebeuren 3 dingen
                result=result.append(tussenres)
                result=result.reset_index(drop=True)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["offset"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["offset"],"done":["no"],"time":[datetime.now()]})])
        nieuweID=nieuweID+1
    #result.iloc[max(dataset["Id"]):,9:]  >0 =0      
    return result,meta

            
            
###################################################     

import timeit

start = timeit.default_timer()
st=datetime.now()
#Your statements here

 
result,meta=random_augmentation(maldiData,300,horizontal_values=[2,25],
                    distribution_augmentation={"horizontal":[40],
                               "add_noise":[20],
                               "linear_line":[20],
                               "multiplication":[20],
                               "offset":[20]})

stop = timeit.default_timer()
end=datetime.now()
print('Time: ', stop - start)          
#  891


end=st







meta["time_shifted"]=meta["time"].shift(1)


meta["diff"]=meta["time"]-meta["time_shifted"]






meta["time"].shift(1)



import cProfile
import re
cProfile.run('re.compile("foo|bar")')



verslag=cProfile.run('random_augmentation(maldiData)')











