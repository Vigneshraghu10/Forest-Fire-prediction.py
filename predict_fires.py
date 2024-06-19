import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import pickle

DATA_FILE='data/forestfires.csv'
MODEL='model.pickle'

df=pd.read_csv(DATA_FILE)
data=df.values

data_y=[]
data_x=[]

for i in range(len(data)):
	data_x.append(data[i][:12])
	data_y.append(data[i][12])

data_x=np.asarray(data_x)
data_y=np.asarray(data_y)

# print(data_x[0])

for i in range(len(data_x)):

	data_x[i][0]=np.float64(data_x[i][0])
	data_x[i][1]=np.float64(data_x[i][1])

	if(data_x[i][2]=='jan'):
		data_x[i][2]=0.0
	elif(data_x[i][2]=='feb'):
		data_x[i][2]=1.0
	elif(data_x[i][2]=='mar'):
		data_x[i][2]=2.0
	elif(data_x[i][2]=='apr'):
		data_x[i][2]=3.0
	elif(data_x[i][2]=='may'):
		data_x[i][2]=4.0
	elif(data_x[i][2]=='jun'):
		data_x[i][2]=5.0
	elif(data_x[i][2]=='jul'):
		data_x[i][2]=6.0
	elif(data_x[i][2]=='aug'):
		data_x[i][2]=7.0
	elif(data_x[i][2]=='sep'):
		data_x[i][2]=8.0
	elif(data_x[i][2]=='oct'):
		data_x[i][2]=9.0
	elif(data_x[i][2]=='nov'):
		data_x[i][2]=10.0
	else:
		data_x[i][2]=11.0

	if(data_x[i][3]=='mon'):
		data_x[i][3]=0.0
	elif(data_x[i][3]=='tue'):
		data_x[i][3]=1.0
	elif(data_x[i][3]=='wed'):
		data_x[i][3]=2.0
	elif(data_x[i][3]=='thu'):
		data_x[i][3]=3.0
	elif(data_x[i][3]=='fri'):
		data_x[i][3]=4.0
	elif(data_x[i][3]=='sat'):
		data_x[i][3]=5.0
	else:
		data_x[i][3]=6.0

# print(data_x[0][2], data_x[0][3])

data_x=scale(data_x.astype(float))

reg_model=make_pipeline(PolynomialFeatures(5), Ridge())
reg_model.fit(data_x, data_y)
# print('Done')
pickle.dump(reg_model, open(MODEL, 'wb'))
print('Regression score: '+str(reg_model.score(data_x, data_y)))