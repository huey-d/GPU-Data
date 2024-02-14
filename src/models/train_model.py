import time
import os

import pandas as pd
import numpy as np

import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics

import pickle
import xgboost
import mlflow

import tensorflow

from hyperopt.pyll import scope
import hyperopt

### Variables for experiment logging ###



# parameter space for Bayesian optimization
param_space = {'eta': hyperopt.hp.loguniform('eta', -9, 0),
              'gamma': hyperopt.hp.loguniform('gamma', -10, 10),
              'max_depth': scope.int(hyperopt.hp.uniform('max_depth', 1, 30)),
              'min_child_weight': hyperopt.hp.loguniform('min_child_weight', -2, 3),
              'max_delta_step': hyperopt.hp.uniform('max_delta_step', 1, 10),
              'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
              'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
              'lambda': hyperopt.hp.loguniform('lambda', -10, 10),
              'alpha': hyperopt.hp.loguniform('alpha', -10, 10),
              'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 1, 10),
              'grow_policy': hyperopt.hp.choice('grow_policy', ['depthwise', 'lossguide']),
              'max_leaves': scope.int(hyperopt.hp.uniform('max_leaves', 0, 10)),
              'n_estimators': scope.int(hyperopt.hp.uniform('n_estimators', 100, 1000)),
              'eval_metric': hyperopt.hp.choice('eval_metric', ['logloss', 'error'])
              }
### Ends here ###


def read_data(data_path: str = "data",
              file_path: str = "processed/gpu_data.csv"):
    """
    Transforms the dataset into a pandas dataframe, synthesize new feature(s), encode categorical features, and split the dataset into training and testing sets for xgboost and NN models

    Args:
        data_path (str, optional): the path to the data directory. Defaults to 'data'
        file_path (str, optional): the directory and path to our file. Defaults to 'processed/gpu_data.csv'
        encode (True): Encodes categorical features using one hot encoding. Defaults to True

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: dataframes and series containing the encoded features and labels
    """
    # Turn csv file into dataframe
    df = pd.read_csv(os.path.join(data_path, file_path), index_col=0)

    df = df.drop(columns=['productname'], axis=1) # we drop product name because it won't be a useful feature

    df = pd.get_dummies(df, columns=['brand', 'chipmanu']) # One hot Encode the categorical values

    df['surface_area'] = df['length'] * df['width']  * df['height'] # Calculate the rectangular surface area of the GPU

    # Separate dataset to features and label columns
    X = df.drop(columns=['price'], axis=1)
    y = df[['price']]
    
    # Create training and testing sets for NN model
    train_set = df.sample(frac=.9)
    test_set = df.drop(train_set.index)

    train_features = train_set.copy()
    test_features = test_set.copy()

    train_labels = train_features.pop('price')
    test_labels = test_features.pop('price')

    return X.values, y.values, train_features, test_features, train_labels, test_labels


def XGB_regression(X: pd.DataFrame,
                   y: pd.DataFrame,
                   experiment_name: str):
    """
    Trains an xgboost regression model and optimizes model hyperparameters using bayesian optemization

    Args:
        X (pd.DataFrame): Dataframe containing the features
        y (pd.DataFrame): Dataframe containing the labels
        experiment_name(str): Name of the experiment recorded in MLFlow
    """

    mlflow.set_experiment(experiment_name=experiment_name)

    
    start = time.time()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    eval_set = [(X_train, y_train), (X_test, y_test)]


    def train_model(params):

        with mlflow.start_run(nested=True):
            

            start_time = time.time()
            model = xgboost.XGBRegressor(**params, objective='reg:squarederror', early_stopping_rounds=100, n_jobs=-1)
            model.fit(X_train, y_train, verbose=True, eval_set=eval_set)
            run_time = time.time() - start_time
            
            # Make predictions for the regression model
            y_pred = model.predict(X_test)

            # Estimate quality of regressor
            y_pred = [round(value) for value in y_pred]

            # use sklearn.metrics to log metrics onto mlflow
            mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
            rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
            mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
            r2 = sklearn.metrics.r2_score(y_test, y_pred)

            mlflow.log_metric('mse', mse)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('r2', r2)
            
            mlflow.log_metric('run_time', run_time)

            
            return {'status': hyperopt.STATUS_OK, 'loss': rmse, 'model': model}


    trials = hyperopt.Trials()
    best_params = hyperopt.fmin(fn=train_model,
                                space=param_space,
                                algo=hyperopt.tpe.suggest,
                                max_evals=25,
                                trials=trials
                                )            


    model_runtime = time.time() - start
    
    best_trial = trials.average_best_error()
    best_parameters = hyperopt.space_eval(param_space, best_params)

    print("\nBest Hyperparameters: {}".format(best_parameters))
    print("\nBest RMSE Score: {}".format(best_trial))
    print("\nModel runtime: {} seconds \n".format(model_runtime))

    best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

    print(best_model)
    pickle.dump(best_model, open('xgboost.pkl', 'wb'))

def NN_regression_model(training_features,
                        training_labels,
                        testing_features,
                        testing_labels,
                        experiment_name,
                        run_name):    
    
    # Set experiment name
    mlflow.set_experiment(experiment_name=experiment_name)

    # Fit the normalization layer to training data
    norm_layer = tensorflow.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(training_features)

    # Define the model architecture
    model = tensorflow.keras.models.Sequential([
        norm_layer,
        tensorflow.keras.layers.Dense(256, input_dim = training_features.shape[1], activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dense(512, activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dense(512, activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dense(512, activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='Adam', loss='MSE', metrics=['MSE', 'MAE'])

    with mlflow.start_run(run_name=run_name):

        mlflow.tensorflow.autolog()
    
        # Train the model
        model.fit(training_features, training_labels, validation_split=0.2, epochs=101, batch_size = 32, verbose=0)

        # Evaluate the model on the testing data
        model.evaluate(testing_features, testing_labels)

        # Make predictions on the testing data
        predictions = model.predict(testing_features)

        # Log metrics to MLFlow
        mse = sklearn.metrics.mean_squared_error(test_labels, predictions)
        rmse = sklearn.metrics.mean_squared_error(test_labels, predictions, squared=False)
        mae = sklearn.metrics.mean_absolute_error(test_labels, predictions)
        r2 = sklearn.metrics.r2_score(test_labels, predictions)

        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)


if __name__== '__main__':
    X, y, train_features, test_features, train_labels, test_labels = read_data()
    # XGB_regression(X, y, experiment_name = 'XGBOOST_2')
    # NN_regression_model(train_features, train_labels, test_features, test_labels, 'NN', '7')