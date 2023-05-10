import pickle
import pandas as pd
import numpy as np
import argparse
import data_collection_and_cleaning as dc

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def classify_tweets(category, user_ids, num_tweets, model):
    df = collect_and_clean_tweets(category, user_ids)
    df["label"] = model.predict(df["text"])
    return df

# The function to start the script, including the whole process of the workflow.
def start(ids, category, max_num_tweets_to_view, mode):

    # Initialize the dataset (tweets) and model
    tweets, model = init(category, ids, mode)
    tweets = tweets_classification(model, tweets)

    # Workflow: push the tweet and receive feedback
    for i in range(0, max_num_tweets_to_view):
        tweets, model = push_tweet_and_get_input(model, tweets)

    # Retrain the model again
    model = retrain(model, tweets)
    tweets = tweets_classification(model, tweets)
    
    # Save the labelled tweets and the trained model
    tweets.to_csv("datasets/" + category + "_labelled.csv")
    pickle.dump(model, open("models/" + category + ".sav", 'wb'))

    print("End train....")

# The function for program initialization
def init(category, ids, mode, clf):
    print("Starting to collect tweets from the entered users... ...")

    # Init according to the mode selected. 
    # A normal mode is for normal data gathering and cleaning.
    # The stub mode is for loading cleaned data for testing. (Avoid call Tweepy too often.)
    if mode != 'normal':
        df = collect_and_clean_tweets(category, ids)
    else:
        # stub
        df = collect_and_clean_tweets_stub(category)
    
    # Init database and model
    df = init_database(df)
    model = init_model(df, clf)

    return df, model

# Tweets collection and cleaning.
def collect_and_clean_tweets(category, ids):
    dc.collect_tweets(category, ids)
    df = dc.clean_tweets(category)
    return df

# Tweets collection and cleaning. (Stub)
def collect_and_clean_tweets_stub(category):
    # direct load from datasets
    df = pd.read_csv("datasets/" + category + ".csv")
    return df

# Init database (dataframe)
def init_database(df):
    df["pred"] = np.random.randint(0,2, size=len(df))
    df["label"] = None
    return df

# Init model for text classification.
def init_model(df, clf):
    
    # create an empty model
    text_clf = Pipeline([
     ('vect', CountVectorizer()),     # converts text into feature vectors (see bag-of-words)
     ('tfidf', TfidfTransformer()),   # converts to frequencies + applying weighting to certain tokens (see tf-idf)
     ('clf', clf),                    # model, we can replace this with whatever model we want
    ])
    
    # Train with init dataset df
    x = df["text"]
    y = df["pred"]

    text_clf.fit(x, y)

    return text_clf

# The function for tweets classification according to the current model
def tweets_classification(model, tweets):
    # classify tweets based on the current model
    predicts = model.predict(tweets["text"])
    tweets["pred"] = predicts

    return tweets

def get_tweet(model, tweets):
    unlabelled_tweets = tweets[tweets['label'].isnull()]
    return unlabelled_tweets[unlabelled_tweets['pred'] == 0]

def push_tweet_and_get_input(model, tweets, response):
    # push the first tweets unlabelled and predicted as 0
    unlabelled_tweets = tweets[tweets['label'].isnull()]
    unlabelled_tweets_0 = unlabelled_tweets[unlabelled_tweets['pred'] == 0]

    push_tweet = unlabelled_tweets_0.iloc[0]
    
    if response == "Yes":
        tweets = confirm(tweets, push_tweet)
    elif response == "No":
        tweets = wrong(tweets, push_tweet)
    else:
        model = retrain(model, tweets)
        tweets = tweets_classification(model, tweets)
    
    return tweets, model

# Option 1
def confirm(tweets, push_tweet):
    target_id = push_tweet['id']
    tweets.loc[tweets['id'] == target_id,'label'] = 0
    
    return tweets

# Option 2
def wrong(tweets, push_tweet):
    target_id = push_tweet['id']
    tweets.loc[tweets['id'] == target_id,'label'] = 1

    return tweets

# Option 3
def retrain(model, tweets):
    print("Retrain the model... ...")
    tweets_labelled = tweets[~tweets['label'].isnull()]
    x = tweets_labelled["text"]
    y = tweets_labelled["label"]
    y=y.astype('int')

    model.fit(x, y)
    return model

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument("-m", "--mode", action='store_true')

    # Parse the argument
    args = parser.parse_args()

    mode = None
    if args.mode == True:
        mode = 'stub'
    else:
        mode = 'normal'

    # Start the script
    start(mode)