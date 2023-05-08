# Provide the token here, replace it with your token in your account
import tweepy
import re
import pandas as pd
import emoji

client = tweepy.Client(bearer_token = "AAAAAAAAAAAAAAAAAAAAAOqYlAEAAAAAmYjJKy1sb6pcRIj7YQbu%2Bz7Me80%3DjGIBoQOZnQqqUKVBHhHhU8zlZsqRftNTXBbYmicJnpYeRniEI7")

def get_userid_from_handle(handle):
    twitterid = client.get_user(username=handle[1:])
    return twitterid.data.id

def match_full_text(text, retweets):
    prefix = re.findall("RT @\w+: ", text)
    incomplete_content = text.strip(prefix[0]).strip("…")
    full_text = None
    for retweet in retweets:
        retweet_content = retweet.text
        if incomplete_content in retweet_content:
            full_text = retweet_content
            break

    if full_text == None:
        return text
    else:
        full_text = prefix[0] + full_text
    
    return full_text

# Remove emoji
def remove_emojis(text):
    return emoji.demojize(text, delimiters=(' ', ' '))

def collect_tweets(category, ids):
    
    filename = "temp/" + category + ".csv"
    dfs = []
    for id in ids:
        response = client.get_users_tweets(id, max_results=100, tweet_fields=['lang', 'author_id', 'referenced_tweets'], expansions=['referenced_tweets.id'])

        if response.data is None:
            continue

        output = []
        for tweet in response.data:
            if tweet.lang == 'en':
                tweet_id = tweet.id
                author_id = tweet.author_id
                text = tweet.text
                is_retweet = True if text.startswith("RT @") == True else False
                
                line = {'id' : tweet_id,'author_id' : author_id, 'text' : text, 'is_retweet': is_retweet}
                output.append(line)

        includes = response.includes

        try:
            reference_tweets = includes['tweets']

            # Match the tweets' incomplete content with the its corresponding source tweets
            for tweet in output:
                id = tweet["id"]
                text = tweet["text"]
                is_retweet = tweet["is_retweet"]
                if is_retweet == True:
                    full_text = match_full_text(text, reference_tweets)
                    tweet['text'] = full_text
        except KeyError as ke:
            # print("No retweets!")
            pass

        df = pd.DataFrame(output).drop_duplicates()
        dfs.append(df)
    
    df_all = pd.concat(dfs, axis=0)
    df_all.to_csv(filename)

    return df_all

def clean_tweets(category):
    df = pd.read_csv("temp/" + category + ".csv")

    # Copy original text
    df['content'] = df['text']

    # Remove urls
    df['text'] = df['text'].str.replace(r'(https?://[^\s]+)', '')

    # use str.replace with a regular expression to remove '@' and the accompanying text
    df['text'] = df['text'].str.replace(r'@\S+\s*', '')
    df['text'] = df['text'].str.replace('RT', '')

    # define a regular expression pattern to match special characters
    special_char_pattern = r'[^a-zA-Z0-9\s]'

    # apply the pattern to the 'text' column using str.replace()
    df['text'] = df['text'].str.replace(special_char_pattern, '')

    # apply the function to the 'text' column of the DataFrame
    df['text'] = df['text'].apply(remove_emojis)

    # save the DataFrame to a CSV file - filename 
    df.to_csv("datasets/" + category + ".csv", index=False)

    return df

# Example
# folder = "uclAI/"
# ids = ["18481648", "4860090466", "337630337", "65722733", "712926286591762433", "244168999", "207581304", "1333768694112129026", "2427400058", "697785916623036416", "277622247", "1192024076006699008", "1107588313509253120"]
# 18481648 4860090466 337630337 65722733 712926286591762433 244168999 207581304 1333768694112129026 2427400058 697785916623036416 277622247 1192024076006699008 1107588313509253120