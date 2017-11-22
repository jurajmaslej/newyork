import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class DataLoader:
	
	def __init__(self, filename):
				self.df = pd.read_csv(filename, header=0, sep=',', names =['vendor', 't_pickup','t_dropoff','passngr_num', 'dst', 'ratecode', 'store_and_fwd', 'loc_pickup', 'loc_dropoff', 'payment_type', 'fare', 'extra', 'tax', 'tip', 'tolls', 'imprv_sur', 'total_price'],
                  dtype = None)
		
	def drop_non_creditcard(self):
		# tips only recorded if payment_type was credit card, thus dropping
		# non credit card payments
		self.df = self.df[self.df.payment_type == 1]
	
	def drop_tax_paid(self):
		# research purpouse only, not used now
		self.df = self.df[self.df.tax == 0]
		print(self.df.shape)
		
	def	data_for_linear(self):
		#add featrues
		# getting best results with these features
		x = self.df[['dst']].values
		x = x.reshape(-1,1)
		y = self.df.tip.values.reshape(-1, 1)
		
		
		split = 350
		x_train = x[:split]
		x_test = x[split:]
		
		y_train = y[:split]
		y_test = y[split:]
		
		return {
			'xtrain': x_train,
			'ytrain': y_train, 
			'xtest' :x_test,
			'ytest': y_test
		}
		
	def data_for_linear_multifeatures(self):
		#add featrues
		# getting best results with these features
		x = self.df[['dst', 'total_price', 'passngr_num', 'tolls']].values
		y = self.df.tip.values.reshape(-1, 1)
		
		
		split = 350
		x_train = x[:split]
		x_test = x[split:]
		
		y_train = y[:split]
		y_test = y[split:]
		
		return {
			'xtrain': x_train,
			'ytrain': y_train, 
			'xtest' :x_test,
			'ytest': y_test
		}
		
	def linear_model(self, data):
		regr = linear_model.LinearRegression()
		
		regr.fit(data['xtrain'], data['ytrain'])
		test_pred = regr.predict(data['xtrain'])
		
		# The mean squared error
		print("Mean squared error test: %.2f"
			% mean_squared_error(data['ytrain'], test_pred))
		print("Mean absolute error test: %.2f"
			% mean_absolute_error(data['ytrain'], test_pred))
		
		# Make predictions using the testing set
		price_pred = regr.predict(data['xtest'])
		
		s_indexes = np.argsort(data['xtest'].ravel())
		
		# The coefficients
		
		print('Coefficients: \n', regr.coef_, regr.intercept_)
		# The mean squared error
		print("Mean squared error: %.2f"
			% mean_squared_error(data['ytest'], price_pred))
		print("Mean absolute error: %.2f"
			% mean_absolute_error(data['ytest'], price_pred))
		# Explained variance score: 1 is perfect prediction
		print('Variance score: %.2f' % r2_score(data['ytest'], price_pred))
		
		
		
		#x_indexes = np.argsort(data['xtest'].ravel())
		# Plot outputs
		x_axis = [i for i in range(data['xtest'].shape[0])] 
	
		plt.scatter(x_axis, data['ytest'],  color='black')
		plt.scatter(x_axis, price_pred, color='red')
		
		
		#plt.plot(data['xtest'].ravel()[s_indexes], price_pred[s_indexes], color='blue', linewidth=3)

		plt.xticks(())
		plt.yticks(())

		plt.show()
			
datas = DataLoader('yellow_tmp.csv')
datas.drop_non_creditcard()

#datadict = datas.data_for_linear()
#datas.linear_model(datadict)

datadict = datas.data_for_linear_multifeatures()
datas.linear_model(datadict)







