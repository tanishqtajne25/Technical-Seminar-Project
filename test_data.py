import sys; sys.path.insert(0, '.')
import tensorflow as tf
from utils import load_data

try:
    frames, alignments = load_data(tf.convert_to_tensor(r'D:\TS\ypc\data\s1_processed\bbaf2n.mpg'))
    print('frames shape:', frames.shape)
    print('alignments shape:', alignments.shape)
    print('SUCCESS')
except Exception as e:
    print('ERROR:', e)
