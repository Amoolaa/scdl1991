import pickle
import os

from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from model_training import classify_tweets
from model_training_runner import ModelTrainingRunner
from data_collection_and_cleaning import get_userid_from_handle
from models import clf_models, clf_model_descriptions

# 18481648 4860090466 337630337 65722733 712926286591762433 244168999 207581304 1333768694112129026 2427400058 697785916623036416 277622247 1192024076006699008 1107588313509253120

# uni example: 
# Input user_ids: monash, unsw, = 15926727 14769030, handles = @UNSW @MonashUni
# input search ids: usyd = 20104025
 
"""
TODO

- fix retrain (DONE)
- take best scoring tweets (NA)
- include description for what each of the models do (tbd)
- tidy up home page 
- tidy up input pages 
- include some info on how good the results were (NA)
"""


app = Flask(__name__)
sess = Session()


@app.route("/")
def home():
  return render_template("home.html")

@app.route("/model_training/submit", methods=['POST', 'GET'])
def submit():
  if request.method == "POST":
    handles = request.form["user_ids"].split(" ")
    ids = [get_userid_from_handle(handle) for handle in handles]
    category = request.form["category"]
    max_tweets = int(request.form["max_tweets"])
    mode = request.form["test"]
    clf_type = request.form["clf_type"]

    if mode == "on":
      mode = "stub"
    else:
      mode = "normal"
    session["model_trainer"] = ModelTrainingRunner(ids=ids, category=category, max_tweets=max_tweets, mode=mode, clf=clf_models[clf_type])
    return redirect(url_for("update_tweet", id=session["model_trainer"].get_latest_tweet()["id"]))
  else:
    return render_template("model_training_submit.html", clf=list(clf_models.keys()), clf_model_descriptions=clf_model_descriptions)

@app.route("/model_training/<id>",  methods=['POST', 'GET'])
def update_tweet(id):
  if session["model_trainer"].max_tweets == 1:
    session["model_trainer"].tweets.to_csv("datasets/" + session["model_trainer"].category + "_labelled.csv")
    pickle.dump(session["model_trainer"].model, open("models/" + session["model_trainer"].category + ".sav", 'wb'))
    return redirect(url_for("successful_train", category=session["model_trainer"].category))
  elif request.method == "POST":
    session["model_trainer"].max_tweets -= 1
    session["model_trainer"].update_tweets_and_model(request.form["submit_button"])
    new_id = session["model_trainer"].get_latest_tweet()["id"]
    return render_template("model_training.html", id=new_id, category=session["model_trainer"].category)
  else:
    return render_template("model_training.html", id=id, category=session["model_trainer"].category)


@app.route("/model_training/success=<category>")
def successful_train(category):
  return render_template("model_training_finished.html", category=category)

@app.route("/classify_tweets", methods=['POST', 'GET'])
def classify():
  num_models = len(os.listdir("models"))
  if num_models == 0:
    return render_template("classify_no_models.html")
  else:
    model_names = [filename[:-4] for filename in os.listdir("models")]
    if request.method == "POST":
      search_name = request.form["search_name"]
      category_name = request.form["model_names"]
      handles = request.form["handles"].split(" ")
      ids = [get_userid_from_handle(handle) for handle in handles]
      num_tweets = int(request.form["num_tweets"])

      loaded_model = pickle.load(open(f"models/{category_name}.sav", 'rb'))
      tweets_df = classify_tweets(search_name, ids, num_tweets, loaded_model)
      classified_tweets_df = tweets_df[tweets_df["label"] == 1]

      print(classified_tweets_df)

      return render_template("feed.html", tweet_ids=classified_tweets_df["id"].to_list()[:num_tweets], search_name=search_name, category=category_name)
    else:
      return render_template('classify_submit.html', model_names=model_names)

if __name__ == "__main__":
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'

  sess.init_app(app)

  app.run()