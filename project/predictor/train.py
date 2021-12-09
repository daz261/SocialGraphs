import networkx as nx
import numpy as np
import pandas as pd
import sklearn.pipeline
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from project.graph_building.build_network import build_network
from project.predictor.prepare_input import prepare_dataset, prepare_dataset_randomly
from project.predictor.prepare_input_success import prepare_dataset_success


def train(pipeline, G):
    X_train, y_train, y_names = prepare_dataset(G, shuffle=True)
    model = pipeline.fit(X_train, y_train)
    return model, y_names


def train_success(pipeline, G):
    X_train, y_train, y_names = prepare_dataset_success(G)
    model = pipeline.fit(X_train, y_train)
    return model, y_names


def visualize_weights(weights, feature_name):
    plt.barh(feature_name, weights)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(left=0.35)
    plt.show()


def evaluate(model: sklearn.pipeline.Pipeline, G, random_dataset=True, success=False, title=""):
    if success:
        X_test, y_test, _ = prepare_dataset_success(G)
    else:
        if random_dataset:
            X_test, y_test, _ = prepare_dataset_randomly(G)
        else:
            X_test, y_test, _ = prepare_dataset(G, shuffle=False)

    y_pred = model.predict(X_test)
    if not success:
        metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()
        f1 = metrics.f1_score(y_test, y_pred, average="macro")
        precision = metrics.precision_score(y_test, y_pred, average="macro")
        recall = metrics.recall_score(y_test, y_pred, average="macro")
        return {"f1": f1,
                "precision": precision,
                "recall": recall}
    else:
        mse = metrics.mean_squared_error(y_test, y_pred)
        corr, p = pearsonr(y_test, y_pred)
        plot_linear_prediction(y_test, y_pred, title)
        return {"mse": mse, "corr": round(corr, 2), "p-value": round(p, 2)}

def plot_linear_prediction(y, lin_predicted, title=""):
    corr, p = pearsonr(y, lin_predicted)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y, lin_predicted, edgecolors=(0, 0, 0), s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', c="red", alpha=0.6)
    ax.set_xlabel('Actual weeks on chart', fontsize=14)
    ax.set_ylabel('Predicted weeks on chart', fontsize=14)
    plt.title(title + "\n Predicted and Actual weeks on chart according to Ridge Regressor \nPearson correlation: " + str(
        np.round(corr, 3)) + ", " + "P-value: " + str(np.round(p, 4)), fontsize=16)
    plt.show()

if __name__ == '__main__':
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LinearRegression())])

    train_range = pd.date_range(start='1/01/1990', end='1/01/1995')
    test_range = pd.date_range(start='1/01/1995', end='1/01/2001')

    G_train = build_network(date_range=train_range, sentiment=True)
    G_test = build_network(date_range=test_range, sentiment=True)
    model, feature_names = train_success(pipeline, G_train)
    train_eval = evaluate(model, G_train, random_dataset=False, success=True)
    test_eval = evaluate(model, G_test, random_dataset=False, success=True)
    visualize_weights(model[1].coef_, feature_names)
    print("TRAIN", train_eval)
    print("TEST", test_eval)
