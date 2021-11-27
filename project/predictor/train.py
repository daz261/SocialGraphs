import networkx as nx
import numpy as np
import pandas as pd
import sklearn.pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from project.graph_building.build_network import build_network
from project.predictor.prepare_input import prepare_dataset, prepare_dataset_randomly


def train(pipeline, G):
    X_train, y_train, y_names = prepare_dataset(G, shuffle=True)
    model = pipeline.fit(X_train, y_train)
    return model, y_names


def visualize_weights(linear_model, feature_name):
    plt.barh(feature_name, linear_model.coef_.tolist()[0])
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(left=0.35)
    plt.show()


def evaluate(model: sklearn.pipeline.Pipeline, G, random_dataset=True):
    if random_dataset:
        X_test, y_test, _ = prepare_dataset_randomly(G)
    else:
        X_test, y_test, _ = prepare_dataset(G, shuffle=False)
    y_pred = model.predict(X_test)
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    f1 = metrics.f1_score(y_test, y_pred, average="macro")
    precision = metrics.precision_score(y_test, y_pred, average="macro")
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    return {"f1": f1,
            "precision": precision,
            "recall": recall}


if __name__ == '__main__':
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression())])
    train_range = pd.date_range(start='1/01/1990', end='1/01/1995')
    test_range = pd.date_range(start='1/01/1995', end='1/01/2001')

    G_train = build_network(date_range=train_range)
    G_test = build_network(date_range=test_range)
    model, feature_names = train(pipeline, G_train)
    train_eval = evaluate(model, G_train)
    test_eval = evaluate(model, G_test)
    visualize_weights(model.steps[1][1], feature_names)
    print("TRAIN", train_eval)
    print("TEST", test_eval)
