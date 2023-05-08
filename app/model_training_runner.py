from model_training import init, tweets_classification, push_tweet_and_get_input

class ModelTrainingRunner:

  def __init__(self, ids, category, max_tweets, mode, clf):
    self.mode = mode
    self.max_tweets = max_tweets
    self.ids = ids
    self.category = category
    self.clf = clf

    self.tweets, self.model = init(category, ids, mode, clf)
    self.tweets = tweets_classification(self.model, self.tweets)

  def get_latest_tweet(self):
    unlabelled_tweets = self.tweets[self.tweets['label'].isnull()]
    unlabelled_tweets_0 = unlabelled_tweets[unlabelled_tweets['pred'] == 0]
    return unlabelled_tweets_0.iloc[0]

  # Updates tweets once
  def update_tweets_and_model(self, response):
    self.tweets, self.model = push_tweet_and_get_input(self.model, self.tweets, response)