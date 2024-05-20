'''
Set of core functions that help with model training and evaluation
'''

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re
import json
import warnings

from pkg_resources import get_distribution
from datetime import datetime, timedelta
from scipy.stats import norm

from sklearn import preprocessing
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample

from numpy import argmax

import mlflow

from .vars import *

MLFLOW_TRACKING_URL = os.getenv("MLFLOW_TRACKING_URL")
GOOGLE_BUCKET = os.getenv("GOOGLE_BUCKET", False)

# from fast ai
# https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py

def make_date(df, date_field):
    '''
    Converts field to date type
    '''
    
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    '''
    Helper function that creates additional date features (ie Is_month_end) based on date_field
    '''
    
    make_date(df, field_name)
    field = df[field_name]
    # prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    prefix = re.sub('[Dd]ate$', '', field_name) if not prefix else prefix
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

def transform(x, function, direction='direct'):
    '''
    Helper function for transofrming a variable
    '''

    functions = TRANSFORM_FUNCTIONS

    f = functions[function][direction]

    return eval(f + "(x)")

def standardize(df_X):
    '''
    Helper function that:
    * standardizes / normalizes numerical features
    * adds extra date related columns to date fields
    * one hot encodes categorical variables
    '''

    # save features

    # transform features and save metadata
    features = []
    
    # standardize numeric features
    numerical_cols = [cname for cname in df_X.columns if df_X[cname].dtype.name in ['int8', 'int64', 'float64']]
    for n in numerical_cols:
        min_x = df_X[n].min()
        max_x = df_X[n].max()
        df_X[n] = (df_X[n] - min_x) / max_x
        features.append({'name':n,'type':df_X[n].dtype.name,'transform':'stdize', 'min':min_x, 'max':max_x})
        
    #features = [{'name':cname,'type':df_X[cname].dtype.name,'transform':'stdize',} for cname in numerical_cols]
    #min_max_scaler = preprocessing.MinMaxScaler()
    #df_X[numerical_cols] = min_max_scaler.fit_transform(df_X[numerical_cols])

    # generate dateparts for dates
    date_cols = [
        cname for cname in df_X.columns 
            if df_X[cname].dtype.name in ['datetime64[ns]', 'datetime64', 'object'] and 
                pd.to_datetime(df_X[cname], errors='coerce', infer_datetime_format=True).notna().any()
    ]
    
    [features.append({'name':cname,'type':df_X[cname].dtype.name,'transform':'dtimes',}) for cname in date_cols]
    for c in date_cols:
        add_datepart(df_X, c, drop=True, prefix=c+'_')

    # one hot encode categorical features
    string_cols = [cname for cname in df_X.columns if df_X[cname].dtype.name in ['category', 'object']]
    df_X[string_cols] = df_X[string_cols].astype('category')
    [features.append({
        'name':cname,
        'type':df_X[cname].dtype.name,
        'transform':'ohe',
        'values':list(df_X[cname].dtype.categories)
    }) for cname in string_cols]
    X = pd.get_dummies(df_X)

    return X, features

def split_dataset(df_reg, selected_features, target, seed=100):
    '''
    Helper function that first standardizes the dataset then splits dataset into train / test
    '''

    df_X = df_reg[selected_features].copy()
    Y = df_reg[target].copy()

    X, features = standardize(df_X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test, features

def balance_dataset(df, target, minor_ind = 1, minority_pct = 0.5):
    '''
    Helper function to downsample an unbalanced dataset
    '''

    # Separate majority and minority classes
    major_ind = (minor_ind - 1) * -1

    df_majority = df[df[target].eq(major_ind)]
    df_minority = df[df[target].ge(minor_ind)]

    minority_samples = len(df_minority.index)

    majority_pct = 1 - minority_pct
    majority_samples = int(majority_pct / minority_pct * minority_samples)

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
        replace=False,                  # sample without replacement
        n_samples=majority_samples,
        random_state=123)               # reproducible results
    
    # Combine minority class with downsampled majority class
    df_reg = pd.concat([df_majority_downsampled, df_minority])
    
    # Display new class counts
    print('rebalanced dataset')
    print(df_reg[target].value_counts())

    # flag records in original dataset as sampled
    df = pd.merge(df, df_reg[target], how='left', left_index=True, right_index=True)

    return df, df_reg

def train_model(algorithm, X_train, y_train, hyperparameters=None):
    '''
    Helper function that trains a model
    '''

    if not(hyperparameters):
        hyperparameters = eval("DEFAULT_" + algorithm + "_HP")
    
    hyperparameter_str = ','.join([k + '=' + ('"' + str(hyperparameters[k]) + '"' if  type(hyperparameters[k]) == str else str(hyperparameters[k])) for  k in hyperparameters])

    model = eval(algorithm + '(' + hyperparameter_str + ')')

    # fit model to training data for current expirement
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, transform_fx=None, title="Model"):
    '''
    Helper function that evaluates a model
    '''

    if type(model).__name__ == 'RandomForestClassifier' or type(model).__name__ == 'GradientBoostingClassifier':

        # calculate pr-curve
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba = [p[1] for p in y_pred_proba]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)

        # locate the index of the largest f score to find optimal threshold
        ix = argmax(fscore)
        threshold = thresholds[ix]
        print('Best Threshold=%f, F-Score=%.3f' % (threshold, fscore[ix]))
        plt.clf()
        plt.plot(thresholds, fscore[:-1])
        plt.show()
        
        # y_pred = model.predict(X_test)
        y_pred = [1 if p>threshold else 0 for p in y_pred_proba]
        
        confusion_matrix =  metrics.confusion_matrix(y_test, y_pred) # Compute confusion matrix to evaluate the accuracy of a classification.
        accuracy =          metrics.accuracy_score(y_test, y_pred)   # Accuracy classification score.
        precision_score =   metrics.precision_score(y_test, y_pred)  # Compute the precision.
        recall_score =      metrics.recall_score(y_test, y_pred)     # Compute the recall.
        # auc = metrics.auc(x, y)                                    # Compute Area Under the Curve (AUC) using the trapezoidal rule.
        f1_score =          metrics.f1_score(y_test, y_pred)         # Compute the F1 score, also known as balanced F-score or F-measure.
        roc_auc_score =     metrics.roc_auc_score(y_test, y_pred_proba)    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1) # Compute Receiver operating characteristic (ROC).
   
        # confusion matrix
        cm_df = pd.DataFrame(confusion_matrix, columns=[['predicted','predicted'],['false','true']],index=[['actual','actual'],['false','true']])
        
        # ROC AUC plot
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score, estimator_name='example estimator')
        display.plot()
        
        # Histogram of probabilities
        yt = pd.DataFrame(y_test).reset_index()
        t = yt.columns[1]
        yt = yt.rename(columns={t:'actual'})
        yp = pd.DataFrame(y_pred_proba, columns=['predicted'])

        df_test = yt.merge(yp, left_index=True, right_index=True)
        v = df_test[df_test['actual'].eq(1)]
        nv = df_test[df_test['actual'].eq(0)]
        
        plt.clf()
        pred_hist_chart, ax = plt.subplots()
        plt.hist([v['predicted'], nv['predicted']], bins=10, density=1, histtype='bar', stacked=True, label=df_test['actual'], rwidth=0.75)
        plt.show()
        # mlflow.log_figure(pred_hist_chart, 'pred_hist_chart.png')
        plt.close()

        # Summarize results
        line_break = (44 + len(title))*'-'
        print('\n')
        print(line_break)
        print("--------------------- " + title + " ---------------------")
        print(line_break)
        print('')
        print('threshold percent:        {:0.2f}'.format(threshold))
        print('')
        print('accuracy:        {:0.2f}'.format(accuracy))
        print('precision score: {:0.2f}'.format(precision_score))
        print('recall score:    {:0.2f}'.format(recall_score))
        print('roc auc score:   {:0.2f}'.format(roc_auc_score))
        print('f1 score:        {:0.2f}'.format(f1_score))
        print('')
        print('Confusion Matrix:')
        print(cm_df)
        print('')
        
        test_metrics = {
            'pred_hist_chart': pred_hist_chart,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'roc_auc_score': roc_auc_score,
            'f1_score': f1_score
        }
        
        
    elif type(model).__name__ == 'RandomForestRegressor':
        if transform_fx:
            y_pred = transform(model.predict(X_test), transform_fx, direction='inverse')
            y_test = transform(y_test, transform_fx, direction='inverse')

            # y_pred = eval(transform_fx + "(model.predict(X_test))")
            # y_test = eval(transform_fx + "(y_test)")
        else:
            y_pred = model.predict(X_test)
        
        errors = abs(y_pred - y_test)
        ape = errors / abs(y_test)
    
        # sets error = pred - 1 when y test is zero
        mape = 100 * np.mean(np.where(y_test.eq(0), y_pred - 1, ape))
    
        # excludes y test = 0
        mape2 = 100 * np.mean(ape[np.isfinite(ape)])
    
        # weighted abs percent error
        wape = 100 * np.sum(errors) / np.sum(abs(y_test))
    
        accuracy = 100 - wape
        transformation_bias = 100 * (np.mean(y_test/y_pred) - 1)
    
        line_break = (44 + len(title))*'-'
        print('\n')
        print(line_break)
        print("--------------------- " + title + " ---------------------")
        print(line_break)
        print('Model Performance with Transformation', transform_fx)
        print('Mean Squared Error (MSE): {:0.2f}'.format(metrics.mean_squared_error(y_test, y_pred)))
        print('Root Mean Squared Error (RMSE): {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
        print('Mean Absolute Percentage Error (MAPE): {:0.2f}'.format(round(mape, 2)))
        print('Weighted Absolute Percentage Error (WAPE): {:0.2f}'.format(round(wape, 2)))
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Transformation Bias: {:0.2f}%.'.format(transformation_bias))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        print('')

        # actual vs predicted
        plt.clf()
        act_v_pred = plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred)
        plt.scatter(y_test, y_test)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()
        
        test_metrics = {
            'act_v_pred': act_v_pred,
            'perf_with_trans': transform_fx,
            'MSE':metrics.mean_squared_error(y_test, y_pred),
            'RMSE':np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
            'MAPE':mape,
            'WAPE':wape,
            'avg_error':np.mean(errors),
            'trans_bias':transformation_bias,
            'accuracy':accuracy
        }
    # feature importance
    importances_df = pd.DataFrame({"feature_names" : model.feature_names_in_, "importances" : model.feature_importances_})
    importances_df = importances_df.sort_values(by=['importances'], ascending=False)
    
    return test_metrics, importances_df, y_pred
    
def save_model(model, X_test, y_test, features, model_type, filename, period_label=None, to_bucket=False):
    '''
    Helper function to save a model to a specified google storage location 
    '''

    # calculate prediction error
    if model_type == 'regression':
        y_pred = model.predict(X_test)
        sum_errs = np.sum(((y_test - y_pred) / y_pred) **2)
        stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    else:
        stdev = None

    results = {}
    results['model'] = model
    results['model_type'] = model_type
    results['stdev'] = stdev
    results['features'] = features
    results['period'] = period_label

    filename = filename + '.pkl'
    modelfile = open('models/' + filename,'wb')
    pickle.dump(results,modelfile)
    modelfile.close()
    
    if to_bucket:
        # save model to Google Storage bucket
        bucket = GOOGLE_BUCKET
        path = 'models/' + filename
        cloud_storage_write(filename, bucket, path)
    
    return 'saved and uploaded'

def save_experiment(model, features, model_metrics, config):
    '''
    Helper function that saves a model to ML Flow
    '''

    # save model
    if not(os.getenv("MLFLOW_TRACKING_URL")):
        return "set MLFLOW_TRACKING_URL"

    MLFLOW_TRACKING_URL = os.getenv("MLFLOW_TRACKING_URL")
    tracking_url = MLFLOW_TRACKING_URL

    mlflow.set_tracking_uri(tracking_url)

    with mlflow.start_run() as model_tracking_run:

        active_run = model_tracking_run.info.run_id
        extra_reqs = f"libs=={get_distribution('libs').version}"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            extra_pip_requirements=[extra_reqs],
        )

        mlflow.set_tag("mlflow.runName", config['experiment_name'])
        mlflow.set_tag("mlflow.note.content", config['experiment_desc'])


        # set test metrics in ML Flow expirement
        #mlflow.log_metric("test_accuracy", metrics['accuracy'])
        #mlflow.log_metric("test_precision", metrics['precision_score'])
        #mlflow.log_metric("test_recall", metrics['recall_score'])
        #mlflow.log_metric("test_roc_auc", metrics['roc_auc_score'])
        #mlflow.log_metric("test_f1_score", metrics['f1_score'])
        # mlflow.log_figure(metrics['pred_hist_chart'], 'pred_hist_chart.png')

        mlflow.log_metric("MSE", model_metrics['MSE'])
        mlflow.log_metric("RMSE", model_metrics['RMSE'])
        mlflow.log_metric("MAPE", model_metrics['MAPE'])
        mlflow.log_metric("WAPE", model_metrics['WAPE'])
        mlflow.log_metric("avg_error", model_metrics['avg_error'])
        mlflow.log_metric("trans_bias", model_metrics['trans_bias'])
        mlflow.log_metric("accuracy", model_metrics['accuracy'])
        mlflow.log_figure(model_metrics['act_v_pred'], 'act_v_pred.png')
    
        config_string = json.loads(json.dumps(config, default=str))
        mlflow.log_dict(config_string, 'model/config.json')

        feature_string = json.loads(json.dumps(features, default=str))
        mlflow.log_dict(feature_string, 'model/features.json')


    return 'Experiment saved to ' + tracking_url + '/#/experiments/0/runs/' + active_run + ' with run_id: ' + active_run

def grid_search(algorithm, X_train, y_train, grid_search_params=None):
    '''
    Helper function that performs grid serach based on either default or provided grid search params
    '''

    warnings.filterwarnings('ignore')

    if not(grid_search_params):
        grid_search_params = DEFAULT_GRID_SEARCH

    # Use the random grid to search and fit best hyperparameters

    rf = eval(algorithm + '()')
    rf_random = RandomizedSearchCV(
        estimator = rf, 
        param_distributions = grid_search_params['param_distributions'], 
        n_iter = grid_search_params['n_iter'], 
        cv = grid_search_params['cv'], 
        verbose = grid_search_params['verbose'], 
        random_state = grid_search_params['random_state'], 
        n_jobs = grid_search_params['n_jobs']
    )

    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_estimator_.get_params()

    return best_params