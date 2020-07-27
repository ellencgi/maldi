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


#from support_1 import *




#
#dataset=maldiData.copy()
#wantednumbersofrecords=300
#horizontal_values=[2,25]
#minpos=2
#maxpos=25
#distribution_augmentation={"horizontal":[40],
#                               "add_noise":[20],
#                               "linear_line":[20],
#                               "multiplication":[20],
#                               "offset":[20]}
#
#
# delen door 100
#


def random_augmentation(dataset,wantednumbersofrecords=300,horizontal_values=[2,25],distribution_augmentation={"horizontal":[40],
                               "add_noise":[20],
                               "linear_line":[20],
                               "multiplication":[20],
                               "offset":[20]}):
    # distributie kans op welke augmentation    
    
    # maak alvast result
    
    
    # meta data bijhouden
    meta=pd.DataFrame.from_dict({"old_ID":[],"new_ID":[],"augmentation":[],"done":[],"time":[]})
    
    nieuweID=dataset["Id"].max()
    
    data=dataset.to_numpy()
    result=np.zeros((data.shape[1],0),dtype="object")  
    
    
    
    
    
    while result.shape[1]<(wantednumbersofrecords-dataset.shape[0]):
        
        #print(result.shape[0])
        #Id = random.sample(range(min(dataset["Id"]), max(dataset["Id"])), 1)[0]
        # welk record pakken we om wat mee te doen?
        Id=random.randint(data[:,0].min(), data[:,0].max())
        
        # welke augmentation
        # gooi met de dobbelsteen om te bepalen wat voor augmentation er gaat gebeuren
        dobbelsteen=random.randint(1,6)
        # maak result dataframe
     
        xvalues=dataset.columns[9:]
        xvalues=[int(x) for x in xvalues]
        
        if dobbelsteen==1:
            if random.randint(0,101)<=distribution_augmentation["horizontal"][0]:
                # doe augmentation
                # horizontal shift

                shiftnr=random.randint(horizontal_values[0],horizontal_values[1])    
                #links of rechts?
                if random.randint(0,1)>0.5:
                    #right
                    alles=np.concatenate((data[Id,0:9],np.zeros(shiftnr),data[Id,(shiftnr+9):]))   
                else:
                    # naar links  
                    alles=np.concatenate((data[Id,0:9],data[Id,(shiftnr+9):],np.zeros(shiftnr))) 
                
                alles[0]=nieuweID
                # nullen moeten nog gefiltert worden
                result=np.concatenate((result,alles.reshape(-1,1)),axis=1)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["horizontal"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["horizontal"],"done":["no"],"time":[datetime.now()]})])       
               
        elif dobbelsteen ==2:
            if random.randint(0,101)<=distribution_augmentation["add_noise"][0]:
                # add noise
                noisevalue=data[Id,9:].std()*0.10
                alles=data[Id,9:]+np.random.normal(-noisevalue,noisevalue,data.shape[1]-9)
                alles[alles<0]=0
                alles=np.concatenate((data[Id,0:9],alles))
                
                alles[0]=nieuweID
                # nullen moeten nog gefiltert worden
                result=np.concatenate((result,alles.reshape(-1,1)),axis=1)
                
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["add_noise"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["add_noise"],"done":["no"],"time":[datetime.now()]})])
            
            
        elif dobbelsteen ==3:
            if random.randint(0,101)<=distribution_augmentation["linear_line"][0]:
                # linear line subtraction
                x2=xvalues[10]
                x1=xvalues[len(xvalues)-1-data[Id,7]]
                y2=data[Id,10]
                y1=data[Id,data.shape[1]-(1+data[Id,7])]
            
                a=(y2-y1)/(x2-x1)
                b= data[Id,10]- a * xvalues[10]
                
                # random linear
                nieuwe_b = np.random.normal(0.5*b,1.5*b)
                # x * a + b = y
                #dataset.columns[10]*nieuwe_a+nieuwe_b=dataset.iloc[Id,10]
                nieuwe_a= (data[Id,10]-nieuwe_b)/xvalues[10]
                #####
                # tel of trek de waardes van de lineare lijn op/af van de data
                y=[nieuwe_a*g+nieuwe_b for g in xvalues]
                
                
                if random.randint(0,1)==0:
                    alles=data[Id,9:]-y
                    alles[alles<0]=0
                    alles=np.concatenate((data[Id,0:9],data[Id,9:]-y))
                else:
                    alles=data[Id,9:]+y
                    alles[alles<0]=0
                    alles=np.concatenate((data[Id,0:9],alles))
                
                
                
                alles[0]=nieuweID
                # nullen moeten nog gefiltert worden
                result=np.concatenate((result,alles.reshape(-1,1)),axis=1)
                
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["linear_line"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["linear_line"],"done":["no"],"time":[datetime.now()]})])
                        
        elif dobbelsteen == 4:
            if random.randint(0,101)<=distribution_augmentation["multiplication"][0]:
                
                alles=np.concatenate(data[Id,0:9],data[Id,9:]*random.uniform(0.90,1.10))
                alles[0]=nieuweID
                # nullen moeten nog gefiltert worden
                result=np.concatenate((result,alles.reshape(-1,1)),axis=1)
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["multiplication"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["multiplication"],"done":["no"],"time":[datetime.now()]})])
            
        elif dobbelsteen ==5:
            if random.randint(0,101)<=distribution_augmentation["offset"][0]:
                
                randomvalue=random.uniform(-0.10,0.10)*data[Id,9:].std()
                alles=data[Id,9:]+randomvalue
                alles[alles<0] = 0
                alles=np.concatenate((data[Id,0:9],alles))
                
                alles[0]=nieuweID
                # nullen moeten nog gefiltert worden
                result=np.concatenate((result,alles.reshape(-1,1)),axis=1)
                
                # houd meta data bij:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[str(nieuweID)],"augmentation":["offset"],"done":["yes"],"time":[datetime.now()]})])
            else:
                meta=pd.concat([meta,pd.DataFrame.from_dict({"old_ID":[str(Id)],"new_ID":[np.NaN],"augmentation":["offset"],"done":["no"],"time":[datetime.now()]})])
        nieuweID=nieuweID+1
        
        result=result.T
         
        result=pd.DataFrame.from_records(result.T,columns=dataset.columns)
        
        result=pd.concat([dataset,result])
               
        
        
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


end-st







meta["time_shifted"]=meta["time"].shift(1)


meta["diff"]=meta["time"]-meta["time_shifted"]






meta["time"].shift(1)



import cProfile
import re
cProfile.run('re.compile("foo|bar")')



verslag=cProfile.run('random_augmentation(maldiData)')


