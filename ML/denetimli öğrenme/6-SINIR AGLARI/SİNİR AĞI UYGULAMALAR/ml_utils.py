import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, tree, ensemble, neural_network, svm, neighbors
from functools import reduce

ADEM_DEFAULTS = dict(
    n_samples1=10000,
    n_samples2=10000,
    C1=[0, 0, 0],
    C2=[1.5, 0, 0],
    r1=5.,
    r2=5.,
    theta1=np.pi / 2,
    theta2=0.4 * np.pi,
    phi1=0.5 * np.pi,
    phi2=1.5 * np.pi,
    n_redundant=20,
    sigma=0.2,
    seed=42,
)

SEKIL_DEFAULTS = dict(
    span=8,
    samples=10000,
    n_redundant=12,
)

SCORING = {
    "accuracy": metrics.make_scorer(metrics.accuracy_score),
    "recall": metrics.make_scorer(metrics.recall_score),
    "precision": metrics.make_scorer(metrics.precision_score),
    "auc": metrics.make_scorer(metrics.roc_auc_score)
}

MODELS = {
    'Decision Tree': tree.DecisionTreeClassifier(),
    'Boosting': ensemble.AdaBoostClassifier(),
    'Neural Network': neural_network.MLPClassifier(),
    'k-NN': neighbors.KNeighborsClassifier(),
    'RBF SVM': svm.SVC(kernel='rbf'),
    'Linear SVM': svm.LinearSVC(),
}

PARAMS = {
    'Decision Tree': {
        'max_depth': [3, 5, 7, 15, None]
        },
    'Boosting': {
        'base_estimator': [tree.DecisionTreeClassifier(max_depth=i) for i in [3, 5, 7, 15, None]],
        'n_estimators': [2, 5, 10, 30, 50]
        },
    'Neural Network': {
        'hidden_layer_sizes': reduce(lambda x,y: x+y, [[(i,)*j for i in range(2,5)] for j in range(1,4)]),
        },
    'RBF SVM': {
        'C': [1e-3, 1e-2, 1e-1, 1., 3.],
        'gamma': [1e-3, 1e-1, 1.]
        },
    'Linear SVM': {
        'C': [1e-3, 1e-2, 1e-1, 1., 3.],
        },
    'k-NN': {
        'n_neighbors': [1, 3, 5, 7, 11, 15]
    }
}



scoring_ = SCORING.copy()
models = MODELS.copy()
params = PARAMS.copy()

def chain_maker(n_samples1, n_samples2, C1, C2, r1, r2, theta1, theta2, phi1, phi2, sigma=0.1, seed=None):
    np.random.seed(seed)
    t1 = np.linspace(-np.pi, np.pi, n_samples1)
    t2 = np.linspace(-np.pi, np.pi, n_samples2)
    # Orthonormal vectors n, u, <n,u>=0
    n1 = np.array([np.cos(phi1) * np.sin(theta1), np.sin(phi1)
                   * np.sin(theta1), np.cos(theta1)])
    u1 = np.array([-np.sin(phi1), np.cos(phi1), 0])
    n2 = np.array([np.cos(phi2) * np.sin(theta2), np.sin(phi2)
                   * np.sin(theta2), np.cos(theta2)])
    u2 = np.array([-np.sin(phi2), np.cos(phi2), 0])
    # P(t) = r*np.cos(t)*u + r*np.sin(t)*(n x u) + C
    P1 = r1 * np.cos(t1)[:, np.newaxis] * u1 + r1 * np.sin(t1)[:, np.newaxis] * \
        np.cross(n1, u1) + C1 + np.random.normal(size=(n_samples1, 3)) * sigma
    P2 = r2 * np.cos(t2)[:, np.newaxis] * u2 + r2 * np.sin(t2)[:, np.newaxis] * \
        np.cross(n2, u2) + C2 + np.random.normal(size=(n_samples2, 3)) * sigma
    return P1, P2


def make_chain(*args, n_redundant=0, **kwargs):
    P1, P2 = chain_maker(*args, **kwargs)
    X = np.concatenate([P1, P2], axis=0)
    y = np.array([0] * P1.shape[0] + [1] * P2.shape[0])
    if n_redundant:
        X = np.concatenate(
            [X, np.random.randn(X.shape[0], n_redundant)], axis=1)
    return X, y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def generate_logit(xx, yy):
    logits = (xx**2 - 2*xx + yy**3 + np.log(xx + 70) - 6 * xx * yy) / 12
    return logits


def generate_probas(xx, yy):
    probas = sigmoid(generate_logit(xx, yy))
    return probas


def make_sekil(span=8, samples=1000, n_redundant=12, random_state=None):
    state = np.random.get_state()
    np.random.seed(random_state)
    X2 = np.random.uniform(-span, span, size=(samples, 2))
    y2 = [np.random.choice([0,1], p=[1-p, p]) for p in generate_probas(*X2.T)]
    if n_redundant:
        X2 = np.concatenate([X2, np.random.uniform(-span, span, size=(X2.shape[0], n_redundant))], axis=1)
    np.random.set_state(state)
    return X2, y2


def compare_accuracy(sonuc1, sonuc2, isim1, isim2):
    fig, ax = plt.subplots(figsize=(7, 4))
    sonuc1['test_accuracy'].plot(logx=True, label=isim1, ax=ax)
    sonuc2['test_accuracy'].plot(logx=True, label=isim2, ax=ax)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f'{isim1} vs {isim2} - Accuracy')
    return fig, ax


def compare_time(sonuc1, sonuc2, isim1, isim2):
    fig, ax = plt.subplots(figsize=(7, 4))
    sonuc1['fit_time'].plot(logx=True, label=isim1, ax=ax, logy=True)
    sonuc2['fit_time'].plot(logx=True, label=isim2, ax=ax, logy=True)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Time (s)')
    ax.set_ylim(1e-6, 10000)
    ax.set_title(f'{isim1} vs {isim2} - Fit Time')
    return fig, ax


def get_evaluation_for_size(model, X, y, param_grid={}, n_splits=10):
    evaluations = []
    for train_size_ in [10, 20, 30, 50, 100, 200, 400, 800, 1600, 3200, 6000, 10000]:
        if 'n_neighbors' in param_grid:
            if train_size_ < 30:
                continue
        cv = \
        model_selection.cross_validate(
            model_selection.GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=3),
            X, y, scoring=scoring_,
            cv=model_selection.ShuffleSplit(n_splits=n_splits, train_size=train_size_),
            return_train_score=True,
            n_jobs=-1
        )
        cv = pd.DataFrame(cv)
    #     cv['train_size'] = train_size_
        evaluations.append(cv.mean().rename(train_size_))
    return pd.concat(evaluations, axis=1).T


def accuracy_vs_sample(sonuclar, isim):
    fig, ax = plt.subplots(figsize=(7, 4))
    df = sonuclar[['train_accuracy', 'test_accuracy']]
    df.columns = ['Train', 'Test']
    df.plot(logx=True, ax=ax)
    ax.grid()
    ax.set_title(f'{isim} - Accuracy vs Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Accuracy')
    return fig, ax

def time_vs_sample(sonuclar, isim):
    fig, ax = plt.subplots(figsize=(7, 4))
    df = sonuclar[['fit_time', 'score_time']]
    df.columns = ['Train', 'Predict']
    df.plot(logx=True, ax=ax, logy=True)
    ax.grid()
    ax.set_title(f'{isim} - Time vs Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylim(1e-6, 10000.)
    ax.set_ylabel('Time (s)')
    return fig, ax


def relu(x):
    return x * (x > 0).astype(float)


PERFECS = {
    'Dataset0': 1.00,
    'Dataset1': 0.9273
}

METFUNCS = {
    'Accuracy': (accuracy_vs_sample, compare_accuracy),
    'Time': (time_vs_sample, compare_time),
}

perfecs = PERFECS.copy()
metfuncs = METFUNCS.copy()
