import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json

import tensorflow as tf
import keras
from tft.tft import *
from demo_functions import *

# Model parameters
model_name = "tft-demo"
with open(model_name+'_meta_params.json', 'r') as json_file:
    params = json.load(json_file)
for key, value in params.items():
    globals()[key] = value


custom_objects = {
    'GenericEmbedding': GenericEmbedding,
    'InterpretableMultiHeadSelfAttention': InterpretableMultiHeadSelfAttention,
    'GatedResidualNetwork': GatedResidualNetwork,
    'VariableSelectionNetwork': VariableSelectionNetwork,
    'GatedLinearUnit': GatedLinearUnit,
    'drop_gate_skip_norm': drop_gate_skip_norm,
    'layerInput': tf.keras.layers.Input,
    'layerLSTM': tf.keras.layers.LSTM,
    'layerTimeDistributed': tf.keras.layers.TimeDistributed,
    'layerDense': tf.keras.layers.Dense,
    'layerDropout': tf.keras.layers.Dropout,
    'layerReshape': tf.keras.layers.Reshape,
    'layerSoftmax': tf.keras.layers.Softmax,
    'layerLayerNormalization': tf.keras.layers.LayerNormalization,
    'layerAttention': tf.keras.layers.Attention,
    'layerEmbedding': tf.keras.layers.Embedding,
    'get_quantile_loss': get_quantile_loss,
    'loss_fn': get_quantile_loss(my_quantiles)
}

with open(model_name+'_input_spec.json', 'r') as json_file:
    input_spec = json.load(json_file)
with open(model_name+'_target_spec.json', 'r') as json_file:
    target_spec = json.load(json_file)

loaded_model = keras.saving.load_model("model_"+model_name+".keras", custom_objects = custom_objects)

# ColumnTransformer is a mess, so better define the inverse transform by hand:
def manual_inverse_trans(pipeline):
  def rev_pipeline(df):
    idf = df.copy()
    idf['date'] = pipeline[0].named_transformers_['extract_date_features'].inverse_transform(idf[['year', 'month', 'day', 'day_of_week']].copy())
    idf['crypto_name'] = pipeline[0].named_transformers_['label_encode_crypto'].inverse_transform(idf[['crypto_id']])[:,0]
    idf[['open', 'high', 'low', 'close']] = pipeline[0].named_transformers_['transform_OHLC'][1].named_transformers_[''].inverse_transform(idf[['open', 'high', 'low', 'close']])
    idf[['open', 'high', 'low', 'close']+['crypto_name']] = pipeline[0].named_transformers_['transform_OHLC'][0].inverse_transform(idf[['open', 'high', 'low', 'close']+['crypto_name']])
    idf['volume'] = pipeline[0].named_transformers_['volume_transformer'].inverse_transform(idf[['volume']])[:,0]
    return idf
  return rev_pipeline

my_pipeline = joblib.load(model_name+'_pipeline.pkl')

target_columns = list(target_spec.keys())
feature_columns = ['volume'] + ['open', 'high', 'low', 'close'] + ['crypto_id'] + ['month', 'day', 'day_of_week']

# Import the data, format and preprocess
df = pd.read_csv("dataset.csv")

df_valid = df[df['date'] >= cutoff_date]
df_valid = my_pipeline.transform(X = df_valid)
df_valid = pd.DataFrame(df_valid, columns = np.insert(feature_columns, [5,6], ['crypto_name', 'year']))

x_valid = df_valid.copy().convert_dtypes()

crypto_names = np.sort(x_valid['crypto_name'].unique())

x_valid_obs_lst = [
  x.iloc[i-tft_lookback:i] 
  for x in [x_valid.loc[x_valid['crypto_name'] == id] for id in crypto_names]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]
x_valid_frc_lst = [
  x.iloc[i:i+tft_lookforward] 
  for x in [x_valid.loc[x_valid['crypto_name'] == id] for id in crypto_names]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]

val_stat_lst = dict(zip(list(input_spec['static'].keys()), [np.array([x[var].iloc[0] for x in x_valid_obs_lst]) for var in list(input_spec['static'].keys())]))
val_obs_lst = dict(zip([ft+'_observed' for ft in list(input_spec['observed'].keys())], [np.array([x[var] for x in x_valid_obs_lst]) for var in list(input_spec['observed'].keys())]))
val_frc_lst = dict(zip([ft+'_forecast' for ft in list(input_spec['forecast'].keys())], [np.array([x[var] for x in x_valid_frc_lst]) for var in list(input_spec['forecast'].keys())]))

val_input_data = val_stat_lst
val_input_data.update(val_obs_lst)
val_input_data.update(val_frc_lst)

val_predictions = loaded_model.predict(
    val_input_data,
    verbose = 2,
    batch_size = batch_size
)


val_pred_array = np.array(val_predictions)

dfs = []
for id in range(len(x_valid_frc_lst)):
  df_pred = pd.DataFrame(val_pred_array[:,id,...].transpose(2, 1, 0).reshape(-1, 4), columns = target_columns)
  df_pred['sample'] = id
  df_pred['quantile'] = np.repeat(my_quantiles, repeats = tft_lookforward)
  df_pred['index'] = np.tile(np.arange(tft_lookforward), len(my_quantiles))
  df_true = x_valid_frc_lst[id].drop(columns=['open', 'high', 'low', 'close'])
  df_true['index'] = np.arange(tft_lookforward)
  df_pred = df_pred.merge(df_true, on='index', how='left')
  dfs.append(df_pred)

result_df = pd.concat(dfs, ignore_index=True)

# result_df.to_csv(model_name+"_raw_predictions.csv")

inv_pipeline = manual_inverse_trans(my_pipeline)

if not ('open' in result_df.columns):
  result_df['open'] = result_df['close']
  result_df['high'] = result_df['close']
  result_df['low'] = result_df['close']

processed_df = result_df.groupby("quantile").apply(inv_pipeline, include_groups = False)
processed_df.to_csv(model_name+"_processed_predictions.csv")




