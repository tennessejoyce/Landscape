import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats.mstats import gmean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from inversions import mergeSort
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde


#Read in the extracted features.
features = pd.read_csv('features.csv',index_col=0)
#Columns associated with content features.
content_features = features.iloc[:,4:]
#Use softmax to compute probabilities.
probabilities = softmax(content_features,axis=1)
#Probability threshold for tagging
threshold = 0.3
#Boolean array of tags.
tags = probabilities>0.3
#Compute the frequency of each tag
frequency = tags.sum(axis=0)
#Plot the most common tags.
#How many tags to show
top_n = 20
best = frequency.sort_values().iloc[-top_n:]
plt.barh(range(top_n),best.values,tick_label=best.index)
plt.xlabel(f'# of posts tagged out of {features.shape[0]} total posts')
plt.title('Most common tags')
plt.tight_layout()
plt.savefig('common_tags.png')
plt.close()
#Compute the average score of each tag.
#avg_score = (features.score * tags).sum(axis=1)/frequency
avg_score = 0*frequency
for name,mask in tags.iteritems():
	if frequency[name]<100:
		avg_score[name] = 0
	else:
		avg_score[name] = gmean(features.score[mask])#.median()
#print(pd.concat([avg_score,frequency],axis=1).sort_values(0))
#Plot the top-scoring tags
top_scoring = avg_score.sort_values().iloc[-top_n:]
plt.barh(range(top_n),top_scoring.values,tick_label=top_scoring.index)
plt.title('Top scoring tags (with >100 samples)')
plt.xlabel(f'Geometric mean score')
plt.tight_layout()
plt.savefig('top_scoring_tags.png')
plt.close()



def sort_criterion(y_pred,y_exact):
	#Convert the score back to an integer, to avoid precision loss.
	integer_exact = np.rint(np.exp(y_exact)-1)
	#Sort the actual scores by their predicted ranking.
	partial_sorted = integer_exact[np.argsort(y_pred)]
	#Count how many pairs are out of order.
	misses = count_inversions(list(partial_sorted))
	#Total number of pairs.
	total = len(y_exact)*(len(y_exact)-1)//2
	#Return the probability that a given pair is out of order.
	return 1-misses/total

def count_inversions(a):
	#Uses merge sort to compute the number of inversions.
	if len(a)==1:
		#Base case for recursion.
		return 0
	#Divide the array in half.
	m = len(a)//2
	left = a[:m]
	right = a[m:]
	#Sort both halves recursively.
	inv_left = count_inversions(left)
	inv_right = count_inversions(right)
	#Merge them back together, overwriting a.
	i=j=k=inv_merge=0
	while i<len(left) and j<len(right):
		if left[i]<=right[j]:
			a[k] = left[i]
			i+=1
		else:
			a[k] = right[j]
			j+=1
			inv_merge += len(left)-i
		k+=1
	a[k:] = left[i:] + right[j:]
	#Return the number of inversions
	return inv_left + inv_right + inv_merge



#Train a model to predict the score, based on ResNet output.
X = content_features.values
y = features.log_score


#Separate into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize the data
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train[:,None])
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.transform(y_train[:,None])[:,0]
y_test = scaler_y.transform(y_test[:,None])[:,0]


#Fit a linear model.
model = LinearRegression()
#Fit to the training data.
model.fit(X_train,y_train)
#Make predictions on the train and test sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
#Calculate the error on the train and test sets.
train_mse = mean_squared_error(y_pred_train,y_train)
test_mse = mean_squared_error(y_pred_test,y_test)
#Print the results
print(f'Train_mse: {train_mse}')
print(f'Test_mse: {test_mse}')
#Calculate the comparison accuracy of the model.
train_acc = sort_criterion(y_pred_train,y_train)
test_acc = sort_criterion(y_pred_test,y_test)
#Print the results
print(f'Train_acc: {train_acc}')
print(f'Test_acc: {test_acc}')

