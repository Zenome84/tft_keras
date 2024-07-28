import tensorflow as tf

def get_quantile_loss(q):
  """
  Given an array of quantiles, define the corresponding loss
  
  The resulting loss function expects lists of tensors, each with a trailing dimension equal to the length of q.
  The resulting loss function expects tensors, where predictions have an extra trailing dimension equal to the length of q. 
  The target data is the same for all quantiles, so a dummy dimension will be added internally for broadcasting.
  """
  
  q = tf.constant(q, dtype=tf.float32)

  def loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    error = tf.subtract(y_pred, tf.expand_dims(y_true, axis = -1))
    loss = tf.math.reduce_mean(tf.math.maximum(tf.math.multiply(q, tf.math.negative(error)), tf.math.multiply(1-q, error)))
    
    return loss

  return loss_fn

