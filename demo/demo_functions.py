# A few functions we need for the demo.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf


def dict_to_dataset(data_dict):
  """
  Convert data that was passed as dictionnary of numpy arrays to a TF Dataset
  """
  
  tensors = {key: tf.convert_to_tensor(value) for key, value in data_dict.items()}
  dataset = tf.data.Dataset.from_tensor_slices(tensors)
  return dataset


# Custom transformer for log transformation and scaling grouped by 'crypto_name'
class GroupedOpenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, grouping, offset=1e-12):
        # Small offset for the log in case price falls to 0
        self.offset = offset
        self.first_open_values = {}
        self.grouping = grouping
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', self.grouping])
        self.first_open_values = df.groupby(self.grouping)['open'].first().to_dict()
        return self
      
    def transform(self, X):
        df = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', self.grouping])
        transformed = df.copy()
        # For instruments already present in the training data, scale using first known value,
        # otherwise, use first value in current data.
        for name, group in df.groupby(self.grouping):
          if name in self.first_open_values:
            first_open_log = np.log(self.first_open_values[name] + self.offset)
          else:
            first_open_log = np.log(df.loc[group.index, 'open'].iloc[0] + self.offset)
          for col in ['open', 'high', 'low', 'close']:
                transformed.loc[group.index, col] = np.log(group[col] + self.offset) - first_open_log
        return transformed.values
      
    def inverse_transform(self, X):
      if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['open', 'high', 'low', 'close', self.grouping])
      transformed = X.copy()
      cols = ['open', 'high', 'low', 'close']
      for name, group in X.groupby(self.grouping):
        transformed.loc[group.index, cols] = (self.first_open_values[name] + self.offset)*np.exp(group[cols]) - self.offset
      return transformed.values


# Extract features we'll need from date
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a 1D array of timestamps
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[:, 0]
        dates = pd.to_datetime(X)
        return np.c_[dates.dt.year, dates.dt.month - 1, dates.dt.day - 1, dates.dt.dayofweek]
      
    def inverse_transform(self, X):
      if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns = ['year', 'month', 'day', 'day_of_week'])
      X[['month', 'day']] = X[['month', 'day']]+1
      X = pd.to_datetime(X[["year", "month", "day"]])
      return X.values
