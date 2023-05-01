from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from model_training import ModelTrainingRunner
import pickle

# 18481648 4860090466 337630337 65722733 712926286591762433 244168999 207581304 1333768694112129026 2427400058 697785916623036416 277622247 1192024076006699008 1107588313509253120

app = Flask(__name__)
sess = Session()

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/model_training/submit", methods=['POST', 'GET'])
def submit():
  if request.method == "POST":
    ids = request.form["user_ids"].split(" ")
    category = request.form["category"]
    max_tweets = int(request.form["max_tweets"])
    mode = request.form["test"]
    if mode == "on":
      mode = "stub"
    else:
      mode = "normal"
    session["model_trainer"] = ModelTrainingRunner(ids=ids, category=category, max_tweets=max_tweets, mode=mode)
    return redirect(url_for("update_tweet", id=session["model_trainer"].get_latest_tweet()["id"]))
  else:
    return render_template("submit.html")

@app.route("/model_training/<id>",  methods=['POST', 'GET'])
def update_tweet(id):
  if session["model_trainer"].max_tweets == 0:
    session["model_trainer"].tweets.to_csv("datasets/" + session["model_trainer"].category + "_labelled.csv")
    pickle.dump(session["model_trainer"].model, open("models/" + session["model_trainer"].category + ".sav", 'wb'))
    return redirect(url_for("classify"))
  elif request.method == "POST":
    session["model_trainer"].max_tweets -= 1
    session["model_trainer"].update_tweets_and_model(request.form["submit_button"])
    new_id = session["model_trainer"].get_latest_tweet()["id"]
    return render_template("temp.html", id=new_id, category=session["model_trainer"].category)
  else:
    return render_template("temp.html", id=id, category=session["model_trainer"].category)

@app.route("/classify_tweets", methods=['POST', 'GET'])
def classify():
  return "Hello World!"

if __name__ == "__main__":
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'

  sess.init_app(app)

  app.run()