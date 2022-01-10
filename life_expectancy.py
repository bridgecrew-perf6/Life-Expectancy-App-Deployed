
import pandas as pd
import numpy as np
import os
import joblib

loaded_model = joblib.load('finalized_model.sav')

def Loaded_model(temp):
    temp1=[]
    for i in temp:
        temp1.append(float(i))
    ip=np.array(temp)
    ip=np.reshape(ip,(-1,1)).T
    ip=ip.astype(np.float64)
    r=loaded_model.predict(ip)
    return round(r[0],2)

