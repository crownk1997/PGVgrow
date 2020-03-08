import tensorflow as tf
from glob import glob
import os
import numpy as np
import data_tool


def inferenceResolution(tfrecord_dir):
    assert os.path.isdir(tfrecord_dir)
    tfr_files = sorted(glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    assert len(tfr_files) >= 1
    tfr_shapes = []
    for tfr_file in tfr_files:
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shapes.append(data_tool.parse_tfrecord_np(record).shape)
            break
    max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
    resolution = max_shape[0]

    assert max_shape[-1] in [1, 3]
    num_channels = max_shape[-1]

    if resolution <= 128:
        num_features = 256
    else:
        num_features = 128
    return num_channels, resolution, num_features


################################ visualizing image grids ################################

def montage(images, grid):

    s = np.shape(images)
    assert s[0] == np.prod(grid) and np.shape(s)[0] == 4
    bigimg = np.zeros((s[1]*grid[0], s[1]*grid[1], s[3]), dtype=np.float32)

    for i in range(grid[0]):
        for j in range(grid[1]):
            bigimg[s[1] * i : s[1] * i + s[1], s[1] * j : s[1] * j + s[1]] += images[grid[1] * i + j]

    return np.rint(bigimg*255).clip(0, 255).astype(np.uint8)

################################ pre-processing real images ################################

def downscale(img):
    s = img.shape
    out = np.reshape(img, [-1, s[1]//2, 2, s[2]//2, 2, s[3]])
    return np.mean(out, axis=(2, 4))

def upscale(img):
    return np.repeat(np.repeat(img, 2, axis=1), 2, axis=2)

def process_real(x, lod_in):
    y = x / 127.5 - 1
    alpha = lod_in - np.floor(lod_in)
    y = (1 - alpha)*y + alpha*upscale(downscale(y))
    for i in range(int(np.floor(lod_in))):
        y = upscale(y)
    return y

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)


def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s // 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s // (60*60), (s // 60) % 60, s % 60)
    else:              return '%dd %02dh %02dm'    % (s // (24*60*60), (s // (60*60)) % 24, (s // 60) % 60)