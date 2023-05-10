from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

estimators = [
    ("DT", DecisionTreeClassifier(criterion='gini', splitter='best', random_state=3)),
    ("SVM", svm.SVC(kernel='sigmoid', C=2, random_state=5, degree=3, decision_function_shape='ovr')),
    ("MLP", MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='lbfgs', random_state=42, max_iter=200))
]

clf_models = {
    "NB": MultinomialNB(alpha = 0.9, fit_prior = False),
    "LR": LogisticRegression(penalty='none', C=5, fit_intercept=False, max_iter=1000),
    "DT": DecisionTreeClassifier(criterion='gini', splitter='best', random_state=3),
    "SVM": svm.SVC(kernel='sigmoid', C=2, random_state=5, degree=3, decision_function_shape='ovr',max_iter=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='lbfgs', random_state=42, max_iter=200), # lbfgs has better convergence on relatively small datasets.
    "RF": RandomForestClassifier(bootstrap = True, max_depth= 7, max_features = 'sqrt', min_impurity_decrease =0.01, min_samples_leaf = 1, min_samples_split= 2, n_estimators = 150, oob_score = True),
    "HYBRID": VotingClassifier(estimators)
}

clf_model_descriptions = {
    "NB": "Naive Bayes",
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
    "SVM": "Support Vector Machines (choose for precision, least unlikely to include irrelevant tweets)",
    "MLP": "Multilayer Perceptron",
    "RF": "Random Forest",
    "HYBRID": "Hybrid model using DT, SVM and MLP (choose for recall, most likely to include irrelevant tweets)"
}