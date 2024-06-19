import numpy as np
import pickle

MODEL='model.pickle'

model=pickle.load(open(MODEL, 'rb'))

day=0
month=0
rh=0
rain=0
wind=0
temp=0

ffmc=92.5
dmc=88
dc=698.6
isi=7.1

data_x=[7, 5, month, day, ffmc, dmc, dc, isi, temp, rh, wind, rain]

data_x=scale(data_x.astype(float))

out=model.predict(data_x)
print(out)