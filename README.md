# Landscape
The goal of this project was to create and analyze a dataset of landscape images scraped from reddit, mostly for curiosity about what kinds of images redditors are posting and upvoting.
The repository contains three python scripts (download_images.py, image_recognition.py, and analyze.py) which handle different aspects of the project. This readme will describe the purpose of each of these scripts, and the conclusions drawn from the data analysis.

## download_images.py
This script uses the Pushift Python API (https://github.com/pushshift/api) to retrive information about landscape images posted to the subreddit r/earthporn.
The script also saves additional information like the title of the post, the username of the redditor who posted it, the time when it was posted, and the score of the post (# of upvotes - # of downvotes).
I used this script to collect all landscape images posted to r/earthporn in 2019, excluding those with a score of 1 or less (which are likely to be missing data or spam posts).
This amounts to 36,760 images.

## image_recognition.py
This script determines the content of the landscape images downloaded from reddit using the pretrained Places365 ResNet50 architecture (https://github.com/CSAILVision/places365).
The convolutional neural network classifies the (previously unlabeled) images into 365 different types of locations, like mountain, lake, forest, etc.

## analyze.py
After downloading and classifying the images, I performed an exploratory analysis on the dataset, using the code in analyze.py.
First, I reduce the continuous probablities into discrete tags by setting a tagging threshold at 30%.
For example, if a given image is predicted to be a mountain with >30% confidence, I tag it as a mountain.
Each image can have anywhere from 0 to 3 tags.
The most common tags are shown in the first figure.
It makes sense that scenic locations like waterfalls and snowy mountains are popular to post.

![common_tags.png](https://github.com/tennessejoyce/Landscape/blob/master/common_tags.png)

The next figure shown the top scoring tags, which is surprisingly quite different from the most common.
To curb the effect of outliers, I've used the geometric mean to compute the average score of each class.
I also disqualified classes with fewer than 100 images. 

![top_scoring_tags.png](https://github.com/tennessejoyce/Landscape/blob/master/top_scoring_tags.png)

Lastly, I used a linear regression model, as implemented in scikit-learn, to attempt to predict the score using the output features of the convolutional neural network.
Because both the scores and probabilitities have widely varying orders of magnitude, I used the raw output of the network (i.e. before applying the softmax to get probabilities) and the logarithm of the score to do this linear regression.
I reserved 20% of the images (selected at random) for a test set, which was not used in the training phase.
In the third figure above, I plot the joint distribution of actual and predicted scores for that test set.
The fit has an R^2 value of 0.22 on the test set (0.24 on the training set), suggesting that there are other important factors and/or randomness involved in the reddit score of an image.
I also tried other machine learning models such as Random Forest regression and a small fully-connected neural network (both implemented in scikit-learn as well), but these actually performed worse than the linear regression model, even after much parameter tuning.

![actual_vs_predicted.png](https://github.com/tennessejoyce/Landscape/blob/master/actual_vs_predicted.png)

A possibly more useful metric than R^2 is the comparison accuracy, which answers the following question: given two images randomly selected from the dataset, can the model correctly predict which one has the higher reddit score?
The probabilty that the model predicts this correctly is the comparison accuracy.
This metric can actually be computed in O(nlogn) rather than O(n^2) using a variant of the merge sort algorithm, which I've implemented in analyze.py.
The linear regression model gives a comparison accuracy of 82.4% on the test set (82.9% on the training set), which is significantly better than random guessing (50%).

## Conclusions
Image recognition methods like convolutional neural networks can be powerful tools for drawing conclusions from an unlabeled dataset of images.
Using this approach, we can answer questions like which types of landscape images are most commonly posted, and which are the most popular.
I've also shown that the output of the neural network can be used as a predictor of how well a post will do.
While there are certainly many other factors beyond just the content of the image (such as aesthetics, or a descriptive post title) which determine its success, a content-based predictor is a good start.
In particular, the predictive model developed here was able to predict with 82.4% accuracy which of two candidate images would recieve a higher reddit score.


