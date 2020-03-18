from psaw import PushshiftAPI
import pandas as pd
import numpy as np
from time import time
import os
import urllib
from PIL import Image
from io import BytesIO
import pickle

api = PushshiftAPI()

#Dictionary to store all the post information.
post_info_dict = {}

#Convert UTC into a string
def age_string(utc_time):
	age_in_seconds = int(time()-utc_time)
	seconds = age_in_seconds % 60
	minutes = (age_in_seconds//60) % 60
	hours = (age_in_seconds//3600) % 24
	days = (age_in_seconds//3600)//24
	return f'{days}d {hours}h {minutes}m {seconds}s'

#Only accept urls from reddit and imgur with jpg filetype.
def good_url(s):
	out = s.startswith('https://i.redd.it')
	out = out or s.startswith('https://i.imgur.com')
	out = out and s.endswith('.jpg')
	return out

collect_post_info = False
if collect_post_info:
	#Generator for searching arbitrarily far back.
	gen = api.search_submissions(limit=200000,subreddit='earthporn')
	#Time range to collect posts from.
	min_utc = 1546300800  #Jan 1, 2019
	max_utc = 1577836800  #Jan 1, 2020

	for i,p in enumerate(gen):
		if p.created_utc < min_utc:
			break
		if p.created_utc > max_utc:
			continue
		if not good_url(p.url):
			#Skip because the URL is from an unknown website.
			continue
		if hasattr(p,'removed_by'):
			#Skip because the post has been removed.
			continue
		if p.score<=1:
			#Likely missing data or a spam post.
			continue
		if i%100==0:
			print(f'{i}:  {age_string(p.created_utc)}')
		post_info_dict[p.id] = [p.title,p.author,p.created_utc,p.score,p.url]

	post_info = pd.DataFrame.from_dict(post_info_dict,orient='index',
					columns=['Title','Author','Timestamp','Score','URL'])
	post_info.to_csv('post_info.csv',index_label='Post ID')

else:
	try:
		post_info = pd.read_csv('post_info.csv',index_col='Post ID')
	except:
		print('Could not find post_info.csv. Please set collect_post_info=True to create the file.')
		exit()

#Download an image from "url" and saves it to "filename".
def download_image(url,filename):
	if os.path.exists(filename):
		print('Image %s already exists. Skipping download.' % filename)
		return True
	try:
		image_data = urllib.request.urlopen(url).read()
	except:
		print('Warning: Could not download image.')
		return False
	try:
		pil_image = Image.open(BytesIO(image_data))
	except:
		print('Warning: Failed to parse image')
		return False
	try:
		pil_image = pil_image.resize((224,224),Image.BILINEAR)
		pil_image_rgb = pil_image.convert('RGB')
	except:
		print('Warning: Failed to convert image to RGB')
		return False
	try:
		pil_image_rgb.save(filename, format='JPEG', quality=90)
		return True
	except:
		print('Warning: Failed to save image')
		return False

#Directory to store the images in.
photos_dir = 'photos2/all_images'
download_images = False
if download_images:
	if not os.path.exists(photos_dir):
		os.mkdir(photos_dir)
	#Store the ids of images which are successfully downloaded.
	good_images = []
	#Loop through each post and download the associated image.
	for i,(post_id,row) in enumerate(post_info.iterrows()):
		print(i,post_id)
		filename = f"{photos_dir}/{post_id}.jpg"
		success = download_image(row.URL,filename)
		if success:
			good_images.append(post_id)
	with open('good_images.p','wb') as f:
		pickle.dump(good_images,f)
else:
	with open('good_images.p','rb') as f:
		good_images = pickle.load(f)
