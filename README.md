## Landscape
Create and analyze a dataset of landscape photos scraped from Reddit. The repository contains four python scripts (scrape.py, tag.py, analyze.py, finetune.py) which handle different aspects of the project. This readme will describe the purpose of each of these scripts, and explain how to run them.

#scrape.py
This script uses the Python Reddit API Wrapper (PRAW -- https://praw.readthedocs.io/en/latest/) to scrape landscape photos as they are posted to reddit. In addition to downloading the photos, additional information about the posts is written to post_info.csv including the title of the post, the username of the redditor who posted it, the time when it was posted, and the score of the post (# of upvotes - # of downvotes).


![common_tags.png](https://github.com/tennessejoyce/Landscape/blob/master/common_tags.png)

![top_scoring_tags.png](https://github.com/tennessejoyce/Landscape/blob/master/top_scoring_tags.png)



