import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
import torchvision.datasets
from torch.nn import functional as F
import os
from PIL import Image
from glob import glob
import numpy as np
from time import time
import pandas as pd
import pickle
from sklearn.decomposition import IncrementalPCA

#Collect auxilliary features from the post_info file.
with open('good_images.p','rb') as f:
	good_images = pickle.load(f)
#Sort alphebetically, to match the order of the Pytorch DataLoader.
good_images.sort()
#Look up the associated features.
post_info = pd.read_csv('post_info.csv',index_col='Post ID')
score = post_info.loc[good_images]['Score'].values
target = np.log(score)
timestamp = post_info.loc[good_images]['Timestamp'].values
time_of_day = (timestamp%(60*60*24))/(60*60)
day_of_week = (timestamp%(60*60*24*7))/(60*60*24)
aux_features = np.transpose(np.vstack([score,target,time_of_day,day_of_week]))
aux_features = pd.DataFrame(aux_features,index=good_images,
				columns=['score','log_score','time_of_day','day_of_week'])



compute_color_features = False
if compute_color_features:
	#Objects for loading the images in batches.
	data_path = 'photos2/'
	n_components=256
	def get_RGB(image):
		#Convert to a numpy array, and drop transparency
		rgb_array = np.array(image)[:,:,:3]
		#Flatten
		return np.reshape(rgb_array,(-1))

	dataset = torchvision.datasets.ImageFolder(root=data_path,transform=get_RGB)
	loader = torch.utils.data.DataLoader(dataset,batch_size=n_components*2,num_workers=0,pin_memory=True)




	color_pca = IncrementalPCA(n_components=n_components)

	for batch_number, (data,_) in enumerate(loader):
		print(batch_number)
		if data.shape[0]>=n_components:
			color_pca.partial_fit(data)
	#Save the PCA transformer to a file
	print('Saving to file...')
	with open('pca.p','wb') as f:
		pickle.dump(color_pca,f)
	#Loop again to transform the data.
	print('Applying dimensionality reduction...')
	color_features = []
	for batch_number, (data,_) in enumerate(loader):
		print(batch_number)
		color_features.append(color_pca.transform(data))
	#Combine the separate batches.
	color_features = np.vstack(color_features)
	color_features = pd.DataFrame(color_features,index=good_images[:color_features.shape[0]])
	print(color_features)
	color_features.to_csv('color_features.csv')
else:
	color_features = pd.read_csv('color_features.csv',index_col=0)

compute_content_features = False
if compute_content_features:

	# load the pre-trained weights
	model_file = 'resnet50_places365.pth.tar'
	model = models.__dict__['resnet50'](num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	#Remove the last two layers (Avg pooling and fully connected).
	model = torch.nn.Sequential(*(list(model.children())[:-2]))
	#Move the Resnet model to the GPU.
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)


	#Objects for loading the images in batches.
	data_path = 'photos2/'
	batch_size=8
	# load the image transformer
	transformer = trn.Compose([trn.ToTensor(),trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transformer)
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=0,pin_memory=True)

	n_components = 256
	content_pca = IncrementalPCA(n_components=n_components)

	with torch.no_grad():
		pca_batch = []
		for batch_number, (data,_) in enumerate(train_loader):
			#Move the batch data onto the GPU
			data = data.to(device)
			#Apply the resnet model to extract content features.
			output = model.forward(data)
			#Covert output to a numpy array, and store for pca.
			pca_batch.append(output.cpu().numpy())
			if (batch_size*(batch_number+1)) % (2*n_components)==0:
				#Every few batches, update the incremental PCA.
				#Combine the separate batches.
				pca_batch = np.concatenate(pca_batch,axis=0)
				#Flatten the features (but not the batch dimension)
				pca_batch = np.reshape(pca_batch,(pca_batch.shape[0],-1))
				#Update PCA
				content_pca.partial_fit(pca_batch)
				#Reset the stored data.
				pca_batch = []
	#Save the PCA transformer to a file
	print('Saving to file...')
	with open('pca.p','wb') as f:
		pickle.dump(transformer,f)
	#Loop again to transform the data.
	print('Applying dimensionality reduction...')
	content_features = []
	with torch.no_grad():
		for batch_number, (data,_) in enumerate(train_loader):
			#Move the batch data onto the GPU
			data = data.to(device)
			#Apply the resnet model to extract content features.
			output = model.forward(data)
			output = output.cpu().numpy()
			output = np.reshape(output,(output.shape[0],-1))
			#Apply PCA
			content_features.append(content_pca.transform(output))
	#Combine the separate batches.
	content_features = np.concatenate(content_features,axis=0)
	content_features = pd.DataFrame(content_features,index=good_images[:content_features.shape[0]])
	content_features.to_csv('content_features.csv')
else:
	content_features = pd.read_csv('content_features.csv',index_col=0)
all_features = aux_features.join(color_features)
all_features = all_features.join(content_features,lsuffix='color_',rsuffix='content_')
print(all_features)
all_features.to_csv('features.csv')