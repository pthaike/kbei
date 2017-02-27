#!/bin/python

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
import pdb
from sklearn import grid_search
from clean import *
import time
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import xgboost as xgb

import pdb


_submit = True
testnum = 60
_step = 14
_ahead = 14
_time = 30
_feature_len = 14


def datscater(x, pre_x):
	scaler = StandardScaler()
	scaler.fit(x)
	X_train = scaler.transform(x)
	X_test = scaler.transform(pre_x)
	return X_train, X_tes


def classficationIndex(x, gap):
	m, n = np.shape(x)
	avg = np.mean(x, axis=1)
	num = 0
	xdat = []
	k = 0

	while num < m:
		ix = np.where((avg >= gap*k)&(avg < gap*(k+1)))
		if np.shape(ix)[1] > 0:
			xdat.append(ix)
			num += np.shape(ix)[1]
		k += 1
	return xdat

def getXbyIndex(x, ix):
	return x[ix]

def xgbpredict(x, y, pre_x):
	# x, pre_x = datscater(x, pre_x)
	xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=0)

	dtrain = xgb.DMatrix(xtrain, label = ytrain, missing = -1)
	dvalid = xgb.DMatrix(xvalid, label = yvalid, missing = -1)
	dpre = xgb.DMatrix(pre_x)

	param = {
		'booster':'gbtree',
		'objective':'reg:linear',
		'early_stopping_rounds':500,
		'max_depth':6,
		'silent' : 1,
		'colssample_bytree':0.8,
		'eta':0.02,
		'nthread':10,
		'seed':400
	}

	watchlist = [ (dtrain,'train'), (dvalid,'val')]
	model = xgb.train(param, dtrain, num_boost_round=500, evals=watchlist)

	# model.save_model('xgb.model')
	print 'predict....'
	#predict
	pre_y = model.predict(dpre, ntree_limit=model.best_iteration)
	# printscore(ytest,pre_y)
	return pre_y



def gbdrtrain(x, y, pre_x):
	clf = GradientBoostingRegressor(n_estimators=740, min_samples_leaf = 0.2, min_samples_split = 40, learning_rate=0.1,max_depth=7, random_state=400, loss='ls').fit(x, y)
	p = clf.predict(pre_x)
	return p


def rfrtrain(x, y, pre_x):
	# x, pre_x = datscater(x, pre_x)
	clf = RandomForestRegressor(n_estimators=200,max_leaf_nodes =20, max_depth=6, random_state=400, n_jobs = 6).fit(x, y)
	pred = clf.predict(pre_x)
	return pred




def predict_help(x_complete, f_len):
	print f_len,
	x,y = gettrain(x_complete, f_len)
	print np.shape(x)
	xtest = gettestx(x_complete, f_len)
	if f_len % 2 == 0:
		p = xgbpredict(x, y, xtest)
	else:
		p = gbdrtrain(x, y, xtest)
	# p = gbdrtrain(x, y, xtest)
	return p


def getsingel(data):
	m,n = np.shape(data)
	ret = []
	for i in range(m):
		ret.append([i])
	return ret



def predict(data, label):
	index = classficationIndex(data, 20)
	# index = getsingel(data)
	ret = []
	for ix in index:
		lab = getXbyIndex(label, ix)
		x_complete = getXbyIndex(data, ix)
		n = len(lab)
		pred = np.array([lab]).reshape((n, 1))
		# pdb.set_trace()
		for i in range(_ahead):
			print "-",
			f_len = _feature_len + i
			p = predict_help(x_complete, f_len)
			pred = np.append(pred, p.reshape((n,1)), axis = 1)
			x_complete = np.append(x_complete, p.reshape((n,1)), axis = 1)
		print ""
		if len(ret) == 0:
			ret = pred
		else:
			ret = np.append(ret, pred, axis = 0)
	return ret
	


def submit():
	#data, label = getdata('../pay_last3.csv')
	# data, label = getdata('../pay_last_noavg.csv')
	data, label = getdata('../pay_last_noavg50.csv')
	ret = predict(data, label)
	print np.shape(ret)
	subresult = pd.DataFrame(ret)
	now = time.strftime('%Y%m%d%H%M%S')
	subresult.values[:,12] = subresult.values[:,12] * (4/3)
	subresult.values[subresult.values < 0] = 0
	subresult = np.round(subresult).astype(int)
	subresult.to_csv('../'+now+'.csv', header = False, index = False)


if __name__ == '__main__':
	submit()