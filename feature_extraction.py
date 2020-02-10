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
import re
import matplotlib.pyplot as plt

#Collect auxilliary features from the post_info file.
with open('good_images.p','rb') as f:
	good_images = pickle.load(f)
#Sort alphebetically, to match the order of the Pytorch DataLoader.
good_images.sort()
#Look up the associated features.
post_info = pd.read_csv('post_info.csv',index_col='Post ID')
score = post_info.loc[good_images]['Score'].values
log_score = np.log10(score)
timestamp = post_info.loc[good_images]['Timestamp'].values
time_of_day = (timestamp%(60*60*24))/(60*60)
day_of_week = (timestamp%(60*60*24*7))/(60*60*24)
aux_features = np.transpose(np.vstack([score,log_score,time_of_day,day_of_week]))
aux_features = pd.DataFrame(aux_features,index=good_images,
				columns=['score','log_score','time_of_day','day_of_week'])
#Read in names of categories.
with open('categories_places365.txt','r') as f:
	categories = [s.split()[0][3:] for s in f]


compute_content_features = True
if compute_content_features:

	# load the pre-trained weights
	model_file = 'resnet50_places365.pth.tar'
	model = models.__dict__['resnet50'](num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	#Move the Resnet model to the GPU.
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()


	#Location of the reddit images.
	data_path = 'photos2/'
	#How many images to process at once.
	batch_size=8
	#Transformer to preprocess the images.
	# load the image transformer
	centre_crop = trn.Compose([
	        trn.Resize((256,256)),
	        trn.CenterCrop(224),
	        trn.ToTensor(),
	        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	transformer = trn.Compose([trn.ToTensor(),trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	#Objects for loading the images.
	train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=centre_crop)
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=0,pin_memory=True)

	with torch.no_grad():
		content_features = []
		for batch_number, (data,_) in enumerate(train_loader):
			#Move the batch data onto the GPU
			data = data.to(device)
			#Apply the resnet model to extract content features.
			output = model.forward(data)
			#Covert output to a numpy array and store it.
			content_features.append(output.cpu().numpy())
			print(f'{batch_size*batch_number}/{len(good_images)}')
	#Combine all the batches together.
	content_features = np.concatenate(content_features,axis=0)
	#Convert to a dataframe
	content_features = pd.DataFrame(content_features,index=good_images[:content_features.shape[0]],columns=categories)
all_features = aux_features.join(content_features)
all_features.to_csv('features.csv')
