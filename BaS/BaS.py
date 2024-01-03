from transformers import pipeline
import tensorflow as tf

#Set device to GPU if available
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"


print(device)