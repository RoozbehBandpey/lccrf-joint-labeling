import sklearn_crfsuite
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import time
import os
from os.path import join, exists, split
import pickle


def train(X_train, y_train, crf_iteration, L1_coefficient, L2_coefficient):
    start = time.time()

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=L1_coefficient,
        c2=L2_coefficient,
        max_iterations=crf_iteration,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    print("Training processed in", round(time.time()-start, 2) , ' seconds.')

    return labels, crf



def Predict(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred



def Classification_report_detail(y_test ,y_pred):
    print(metrics.flat_classification_report(y_test, y_pred, digits=4))


def Classification_report_fscore_micro(y_test ,y_pred):
    return metrics.flat_f1_score(y_test, y_pred, average='micro')

def Classification_report_fscore_weighted(y_test ,y_pred):
    return metrics.flat_f1_score(y_test, y_pred, average='weighted')

def Classification_report_fscore_macro(y_test ,y_pred):
    return metrics.flat_f1_score(y_test, y_pred, average='macro')

def Classification_report_accuracy(y_test ,y_pred):
    return metrics.flat_accuracy_score(y_test, y_pred)




def hyperparameter_optimization(X_train, y_train, X_test, y_test, crf_iteration, search_iteration, cv, L1_coefficient, L2_coefficient, task):
    #To improve quality try to select regularization parameters using randomized search and 3-fold cross-validation.
    coefficients = {}
    model_dir = 'RandomizedSearch_hyperparameters'
    model_name = "{0}_{1}_search_iteration_{2}_crossvalidation".format(task, search_iteration, cv)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        hyperparameters_file = open(model_name, "rb")
        hyperparameters = pickle.load(hyperparameters_file)
        hyperparameters_file.close()
        coefficients = hyperparameters['coefficient']
        print('Loading existing hyperparameters \'%s\'' % split(model_name)[-1])
        print("______________________________________________________________________")
        for item in hyperparameters:
            print(item, hyperparameters[item])
    else:
        start = time.time()
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=crf_iteration,
            all_possible_transitions=True
        )

        params_space = {
            'c1': scipy.stats.expon(scale=L1_coefficient),
            'c2': scipy.stats.expon(scale=L2_coefficient),
        }

        crf.fit(X_train, y_train)
        labels = list(crf.classes_)
        y_pred = crf.predict(X_test)

        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

        rs = RandomizedSearchCV(crf, params_space,
                                cv=cv,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=search_iteration,
                                scoring=f1_scorer)
        rs.fit(X_train, y_train)


        hyperparameters = {
            'coefficient': rs.best_params_,
            'best CV score': rs.best_score_,
            'model size': rs.best_estimator_.size_ / 1000000,
            'runtime': round(time.time() - start, 2)
        }

        coefficients = hyperparameters['coefficient']

        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving hyperparameters \'%s\'' % split(model_name)[-1])
        hyperparameters_file = open(model_name, "wb")
        pickle.dump(hyperparameters, hyperparameters_file)
        hyperparameters_file.close()

        print("______________________________________________________________________")
        for item in hyperparameters:
            print(item, hyperparameters[item])

    return coefficients
