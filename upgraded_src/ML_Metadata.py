import numpy as np
import pandas as pd
import argparse
import glob
import time
from utils import IMAGE_CLEF_META, IMAGE_CLEF_KEYS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
import scipy
from scipy.stats import randint as sp_randint

CLF_DICT = {'RF': RandomForestClassifier(), 'SVM': OneVsRestClassifier(SVC())}

def get_samples(report_path, file_ids = None, training_mode = True):
    metadata_col = IMAGE_CLEF_META.keys_as_list()
    report = pd.read_csv(report_path)
    file_ids = report.Filename.values if file_ids is None else file_ids
    mask = report.Filename.isin(file_ids)
    X = report.loc[mask][metadata_col]
    if training_mode:
        tasks_clf = IMAGE_CLEF_KEYS.keys_as_list()
        y = report.loc[mask][tasks_clf]
        y[IMAGE_CLEF_KEYS.SEVERITY] = (y[IMAGE_CLEF_KEYS.SEVERITY] > 3).astype(np.int8)
        return X, y
    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--clf',
        help='',
        choices=[k for k in CLF_DICT.keys()],
        default='SVM',
        type=str
    )

    parser.add_argument(
        "--num_parallel_calls",
        help="Paralle call at Dataset map",
        default=1,
        type=int
    )

    parser.add_argument(
        "--iters",
        help="Paralle call at Dataset map",
        default=1,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__

    X, y = get_samples( '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_metaData_extra_copy.csv')
    print(X)
    print(y)
    print(X.values)
    print(y.values)
    X = X.values
    y = y.values
    print(X.shape, y.shape)
    clf = CLF_DICT[args.clf]
    print(clf)
    if isinstance(clf, RandomForestClassifier):
        param_dist = {"n_estimators":sp_randint(5,500),
                      "max_depth": [3,10, None],
                      "max_features": ["auto", "sqrt", "log2", None],
                      "min_samples_split": sp_randint(2, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
    else:
        param_dist = {'estimator__C': scipy.stats.expon(scale=100), 'estimator__gamma': scipy.stats.expon(scale=.1),
                      'estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'estimator__degree': sp_randint(2,6),
                      'estimator__coef0': scipy.stats.expon(scale=0.0), 'estimator__class_weight': ['balanced', None]}

    # run randomized search
    print(param_dist)
    n_iter_search = args.iters
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, iid=False, n_jobs=args.num_parallel_calls, scoring='f1_micro')

    random_search.fit(X, y)

    print(random_search.cv_results_)
    pd.DataFrame(random_search.cv_results_).to_csv('/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/'+args.clf+'_results_'+ time.strftime("%Y%m%d%H%M%S")+'.csv')
    #print(random_search.best_score_)
    print('Error',random_search.scorer_)
