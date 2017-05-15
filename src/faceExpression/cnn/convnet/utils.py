"""
Utilities for convnet.py
"""

import math
import os
import re
import subprocess

import tensorflow as tf
import tensorflow.contrib.layers as tflayers


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('gpu_memory', 1.0,
                          """the fraction of memory that the process is allowed to use in a gpu""")

base_dir = os.path.abspath(os.path.join(__file__, '../../../'))

__str2activation = {
    'linear': lambda x: x,
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'elu': tf.nn.elu,
    'tan': tf.tanh,
    'sigmoid': tf.sigmoid
}

__str2loss_func = {
    # Currently not supported
    # 'softmax': tf.nn.softmax_cross_entropy_with_logits,
    # 'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits,
    'sparse_softmax': tf.nn.sparse_softmax_cross_entropy_with_logits
}

__str2learning_rate = {
    'exponential': tf.train.exponential_decay,
    'inverse_time': tf.train.inverse_time_decay,
    'natural_exp': tf.train.natural_exp_decay,
    'piecewise_constant': tf.train.piecewise_constant,
    'polynomial_decay': tf.train.polynomial_decay
}


__str2optimizer = {
    'GradientDescent': tf.train.GradientDescentOptimizer,
    'Adadelta': tf.train.AdadeltaOptimizer,
    'Adagrad': tf.train.AdagradOptimizer,
    'AdagradDA': tf.train.AdagradDAOptimizer,
    'Momentum': tf.train.MomentumOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'Ftrl': tf.train.FtrlOptimizer,
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer,
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer
}


def get_activation(str='relu'):
    """
     A utility function to get specified activation function
    :param str: a string which should be among the keys of __str2activation, or a user defined callable function
    :return: a callable function
    """
    if callable(str):
        return str
    if str in __str2activation:
        return __str2activation[str]
    print('No matching activation function found. Using ReLU by default.\n')
    return __str2activation['relu']


def get_loss_func(str='sparse_softmax'):
    """
    A utility function to get specified loss function
    :param str: a string in the keys of __str2loss_func, or a user defined callable function.
        The function is of type: (logits, labels) -> scalar
    :return: a callable loss function
    """
    if callable(str):
        return str
    if str in __str2loss_func:
        return lambda logits, labels: __str2loss_func[str](labels=labels, logits=logits)
    print('No matching loss function found. Using sparse softmax cross entropy by default.\n')
    return __str2loss_func['sparse_softmax']


def get_learning_rate(update_func, **kwargs):
    if callable(update_func):
        return update_func(kwargs['global_step'])
    if update_func == 'piecewise_constant':
        kwargs['x'] = kwargs.pop('global_step')
        kwargs.pop('decay_steps')
        kwargs.pop('learning_rate')
    if update_func in __str2learning_rate:
        return __str2learning_rate[update_func](**kwargs)
    print('No matching learning rate decay scheme found. Using constant scalar learning rate.\n')
    return kwargs['learning_rate']


def get_optimizer(optimizer='Momentum'):
    """
    Use a string to get specific optimizer used for training
    :param optimizer: a string which should be among the keys of __str2optimizer
    :return: a optimizer instance
    """
    if callable(optimizer):
        return optimizer
    if optimizer in __str2optimizer:
        return __str2optimizer[optimizer]
    print('No matching optimizer found. Using Momentum Optimizer by default.\n')
    return __str2optimizer['Momentum']


def get_regularizer(regularizer, scale=0):
    """
    A utility function to get user specified regularizer
    e.g.: get_regularizer('l2', 0.005)
    :param regularizer: of function type (x): (func weights: scalar).
        Take a scalar as input, return a function of type: take a Tensor as input and return a scalar
    :param scale: a scalar
    :return: a function of type: take a list of Tensor as input and return a scalar
    """
    if regularizer is None:
        return lambda x: 0
    _regularizer = _get_regularizer(regularizer, scale)
    return lambda weight_list: tflayers.apply_regularization(_regularizer, weight_list)


def _get_regularizer(regularizer, scale=0):
    if callable(regularizer):
        return regularizer(scale)
    if isinstance(regularizer, list):
        r_list = []
        for reg in regularizer:
            r_list.append(_get_regularizer(reg, scale))
        return tflayers.sum_regularizer(r_list)

    elif regularizer == 'l1':
        return tflayers.l1_regularizer(scale)
    elif regularizer == 'l2':
        return tflayers.l2_regularizer(scale)


def output_shape(input_shape, kernel_shape, strides, padding):
    """
    Given specific conditions calculate the output shape of a convolutional layer
    :param input_shape: input shape, e.g.: [28, 28, 3]
    :param kernel_shape: kernel/filter shape, e.g.: [2,2]
    :param strides: the marching strides shape. e.g.: [1,2,2,1]
    :param padding: a tring indicating padding type. 'SAME' or 'VALID'
    :return: the output shape
    """
    if padding == 'SAME':
        x = math.ceil(input_shape[0] / float(strides[1]))
        y = math.ceil(input_shape[1] / float(strides[2]))
        return x, y

    elif padding == 'VALID':
        x = math.ceil((input_shape[0] - kernel_shape[0] + 1) / float(strides[1]))
        y = math.ceil((input_shape[1] - kernel_shape[1] + 1) / float(strides[2]))
        return x, y


def create_filename(directory):
    """
    Safely create a directory
    :param directory: a string indicating the path
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def before_save(file_or_dir):
    """
    make sure that the dedicated path exists (create if not exist)
    :param file_or_dir:
    :return:
    """
    dir_name = os.path.dirname(os.path.abspath(file_or_dir))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_path(path, file_name=None, absolute=False):
    """
    A helper function that get the real/abs path of a file on disk, with the project dir as the base dir.
    Note: there is no checking on the illegality of the args!
    :param path: a relative path to base_dir, optional file_name to use
    :param file_name: an optional file name under the path
    :param absolute: return the absolute path
    :return: return the path relative to the project root dir, default to return relative path to the called place.
    """
    _p = os.path.join(base_dir, path)
    if file_name:
        _p = os.path.join(_p, file_name)
    if absolute:
        return os.path.abspath(_p)
    return os.path.relpath(_p)


def init_tf_environ(gpu_num=0):
    """
    Init CUDA environments, which the number of gpu to use
    :param gpu_num:
    :return:
    """
    cuda_devices = ""
    if gpu_num == 0:
        print("Not using any gpu devices.")
    else:
        try:
            best_gpus = pick_gpu_lowest_memory(gpu_num)
            cuda_devices = ",".join([str(e) for e in best_gpus])
            print("Using gpu device: {:s}".format(cuda_devices))
        except:
            cuda_devices = ""
            print("Cannot find gpu devices!")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # if FLAGS.gpu_num == 0 else "0,1,2,3"[:(FLAGS.gpu_num * 2 - 1)]


def config_proto():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
    return tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options, allow_soft_placement=True)

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldn't parse "+line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory(num=1):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memorys, best_gpus = list(zip(*sorted(memory_gpu_map)[:num]))
    return best_gpus
