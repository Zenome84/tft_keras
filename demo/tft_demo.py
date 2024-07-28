# This is a simple demo of TFT on a small publicly available crypto dataset.
# It demonstrate the use of tf.Dataset, although in practice for larger datasets
# one would want to use Dataset.from_generator.
# Tested on Python 3.10 with TF-2.15

import numpy as np
import pandas as pd
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
tft_lookback = 32
tft_lookforward = 8
dropout_rate=0.5
d_model=16
att_heads=4
my_quantiles = [0.1, 0.5, 0.9]
n_epochs = 100
batch_size = 32

# Cutoff between training and validation (roughly 2/3 - 1/3)
cutoff_date = "2021-01-01"

params = {
  'model_name':model_name,
  'cutoff_date':cutoff_date,
  'tft_lookback':tft_lookback,
  'tft_lookforward':tft_lookforward,
  'dropout_rate':dropout_rate,
  'd_model':d_model,
  'att_heads':att_heads,
  'my_quantiles':my_quantiles,
  'batch_size':batch_size,
}
with open(model_name+'_meta_params.json', 'w') as json_file:
    json.dump(params, json_file)

with open(model_name+'_input_spec.json', 'r') as json_file:
    input_spec = json.load(json_file)
with open(model_name+'_target_spec.json', 'r') as json_file:
    target_spec = json.load(json_file)



# Activate AMP (currently incompatible with saving/loading serialized model)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Initialize the TemporalFusionTransformer model
tft_model = TemporalFusionTransformer(
    input_spec=input_spec,
    target_spec=target_spec,
    d_model=d_model,
    att_heads=att_heads,
    lookback=tft_lookback,
    lookforward=tft_lookforward,
    dropout_rate=dropout_rate
)

# Apply get_quantile_loss to each output (potentially with different quantiles)
losses = dict(zip(target_spec.keys(), [get_quantile_loss(target_spec[var]['quantiles']) for var in list(target_spec.keys())]))

# Compile the model (I'm scaling losses to track the mean)
tft_model.model.compile(optimizer='adam', loss = losses, loss_weights = 1.0/len(target_spec))


## TEST SAVING/LOADING MODEL/MODEL WEIGHTS
# # Try saving model
# tft_model.model.save("keras_save_files/model_"+model_name+".keras")
# print("Saving model worked!")
# 
# # Try saving weights
# tft_model.model.save_weights("keras_save_files/model_"+model_name+".weights.h5")
# print("Saving weights worked!")
# 
# 
# # Try loading weights:
# tft_model.model.load_weights("keras_save_files/model_"+model_name+".weights.h5")
# print("Loading weights worked!")
# # exit()
# 
# # In order to load the model, we need to pass all custom objects:
# custom_objects = {
#     'GenericEmbedding': GenericEmbedding,
#     'InterpretableMultiHeadSelfAttention': InterpretableMultiHeadSelfAttention,
#     'GatedResidualNetwork': GatedResidualNetwork,
#     'VariableSelectionNetwork': VariableSelectionNetwork,
#     'GatedLinearUnit': GatedLinearUnit,
#     'drop_gate_skip_norm': drop_gate_skip_norm,
#     'layerInput': tf.keras.layers.Input,
#     'layerLSTM': tf.keras.layers.LSTM,
#     'layerTimeDistributed': tf.keras.layers.TimeDistributed,
#     'layerDense': tf.keras.layers.Dense,
#     'layerDropout': tf.keras.layers.Dropout,
#     'layerReshape': tf.keras.layers.Reshape,
#     'layerSoftmax': tf.keras.layers.Softmax,
#     'layerLayerNormalization': tf.keras.layers.LayerNormalization,
#     'layerAttention': tf.keras.layers.Attention,
#     'layerEmbedding': tf.keras.layers.Embedding,
#     'get_quantile_loss': get_quantile_loss,
#     'loss_fn': get_quantile_loss(my_quantiles)
# }
# loaded_model = keras.saving.load_model("keras_save_files/model_"+model_name+".keras", custom_objects = custom_objects)
# print("Loading full model worked!")


# Import the data, format, and preprocess
# Download or import directly from https://www.kaggle.com/datasets/maharshipandya/-cryptocurrency-historical-prices-dataset/data
df = pd.read_csv("dataset.csv", index_col = 0)

target_columns = list(target_spec.keys())
feature_columns = ['volume'] + ['open', 'high', 'low', 'close'] + ['crypto_id'] + ['month', 'day', 'day_of_week']

my_pipeline = Pipeline([
    ('preprocessing', ColumnTransformer([
        ('volume_transformer', Pipeline([
          ('volume_log1p', FunctionTransformer(np.log1p, inverse_func = np.expm1)),
          ('volume_scaler', StandardScaler())
        ]), ['volume']),
        ('transform_OHLC', Pipeline([
          ('grouped_trans', GroupedOpenTransformer(offset = 1e-12, grouping = 'crypto_name')),
          ('scale_OHLC', ColumnTransformer([('', StandardScaler(), [0,1,2,3])], remainder='passthrough'))
        ]), ['open', 'high', 'low', 'close'] + ['crypto_name']),
        ('label_encode_crypto', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = 63), ['crypto_name']),
        ('extract_date_features', DateFeatureExtractor(), ['timestamp'])
    ]))
])

df_train = df[df['date'] < cutoff_date]
df_valid = df[df['date'] >= cutoff_date]

df_train = my_pipeline.fit_transform(X = df_train)
df_valid = my_pipeline.transform(X = df_valid)

joblib.dump(my_pipeline, model_name+'_pipeline.pkl')

# ColumnTransformer returns arrays, so we have to convert back to pandas:
df_train = pd.DataFrame(df_train, columns = np.insert(feature_columns, [5,6], ['crypto_name', 'year']))
df_valid = pd.DataFrame(df_valid, columns = np.insert(feature_columns, [5,6], ['crypto_name', 'year']))


x_train = df_train[feature_columns].convert_dtypes()
y_train = df_train[target_columns].convert_dtypes()

crypto_ids = np.sort(x_train['crypto_id'].unique())

x_train_obs_lst = [
  x.iloc[i-tft_lookback:i]
  for x in [x_train.loc[x_train['crypto_id'] == id] for id in crypto_ids]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]
  
print("Training samples: "+ str(len(x_train_obs_lst)))

x_train_frc_lst = [
  x.iloc[i:i+tft_lookforward] 
  for x in [x_train.loc[x_train['crypto_id'] == id] for id in crypto_ids]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]

y_train_frc_lst = [
  y.iloc[i:i+tft_lookforward]
  for y in [y_train.loc[x_train['crypto_id'] == id] for id in crypto_ids]
  for i in range(tft_lookback,len(y)-tft_lookforward,tft_lookforward)
  ]
  
stat_lst = dict(zip(list(input_spec['static'].keys()), [np.array([x[var].iloc[0] for x in x_train_obs_lst]) for var in list(input_spec['static'].keys())]))
obs_lst = dict(zip([ft+'_observed' for ft in list(input_spec['observed'].keys())], [np.array([x[var] for x in x_train_obs_lst]) for var in list(input_spec['observed'].keys())]))
frc_lst = dict(zip([ft+'_forecast' for ft in list(input_spec['forecast'].keys())], [np.array([x[var] for x in x_train_frc_lst]) for var in list(input_spec['forecast'].keys())]))

# Build dictionary
input_data = stat_lst
input_data.update(obs_lst)
input_data.update(frc_lst)


# Validation data
# We use crypto_name in place of crypto_id for validation otherwise all 
# cryptos appearing in validation but not training would get mixed up.

x_valid = df_valid.copy().convert_dtypes()
y_valid = df_valid[target_columns].convert_dtypes()

crypto_names = np.sort(x_valid['crypto_name'].unique())

x_valid_obs_lst = [
  x[feature_columns].iloc[i-tft_lookback:i] 
  for x in [x_valid.loc[x_valid['crypto_name'] == id] for id in crypto_names]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]
x_valid_frc_lst = [
  x[feature_columns].iloc[i:i+tft_lookforward] 
  for x in [x_valid.loc[x_valid['crypto_name'] == id] for id in crypto_names]
  for i in range(tft_lookback,len(x)-tft_lookforward,tft_lookforward)
  ]
y_valid_frc_lst = [
  y.iloc[i:i+tft_lookforward]
  for y in [y_valid.loc[x_valid['crypto_name'] == id] for id in crypto_names]
  for i in range(tft_lookback,len(y)-tft_lookforward,tft_lookforward)
  ]


val_stat_lst = dict(zip(list(input_spec['static'].keys()), [np.array([x[var].iloc[0] for x in x_valid_obs_lst]) for var in list(input_spec['static'].keys())]))
val_obs_lst = dict(zip([ft+'_observed' for ft in list(input_spec['observed'].keys())], [np.array([x[var] for x in x_valid_obs_lst]) for var in list(input_spec['observed'].keys())]))
val_frc_lst = dict(zip([ft+'_forecast' for ft in list(input_spec['forecast'].keys())], [np.array([x[var] for x in x_valid_frc_lst]) for var in list(input_spec['forecast'].keys())]))

print("Validation samples: "+ str(len(x_valid_obs_lst)))


val_input_data = val_stat_lst
val_input_data.update(val_obs_lst)
val_input_data.update(val_frc_lst)


# Define y for training and validation:
target_data = dict(zip(target_columns, [np.array([y[var] for y in y_train_frc_lst]) for var in target_columns]))
val_target_data = dict(zip(target_columns, [np.array([y[var] for y in y_valid_frc_lst]) for var in target_columns]))

# Prepare data for tf.Dataset:

x_train_ds = dict_to_dataset(input_data)
y_train_ds = dict_to_dataset(target_data)
x_valid_ds = dict_to_dataset(val_input_data)
y_valid_ds = dict_to_dataset(val_target_data)

train_dataset = tf.data.Dataset.zip((x_train_ds, y_train_ds))
valid_dataset = tf.data.Dataset.zip((x_valid_ds, y_valid_ds))

# Shuffle the dataset, batch it, and prefetch for performance (overkill for demo of course)
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_dataset = valid_dataset.shuffle(buffer_size=100)
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


## Fit the model:

# Define callbacks:
cb_cp = keras.callbacks.ModelCheckpoint(
  filepath = "keras_save_files/model_"+model_name+"_checkpoints/model_ep{epoch:02d}.weights.h5",
  save_weights_only = True,
  verbose = 1,
  save_freq = 'epoch'
)
csv_logger = keras.callbacks.CSVLogger("keras_save_files/model_"+model_name+"_history_log.csv", append=True)

# Profile if needed:
# tf.profiler.experimental.start("keras_save_files/model_"+model_name+"_mem_log", options=tf.profiler.experimental.ProfilerOptions(
#     host_tracer_level=3,
#     device_tracer_level=0,
#     python_tracer_level=1,
#     delay_ms=0
# ))

tft_model.model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs=n_epochs,
    # batch_size=batch_size, # not needed when using tf.Dataset
    # shuffle=False, # not needed when using tf.Dataset
    callbacks = [cb_cp, csv_logger]
)

# tf.profiler.experimental.stop()

# Save the trained model:
tft_model.model.save("model_"+model_name+".keras")

