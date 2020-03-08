import tensorflow as tf
from tensorflow.python.ops import nccl_ops
from collections import OrderedDict

def run(*args, **kwargs): # Run the specified ops in the default session.
    return tf.get_default_session().run(*args, **kwargs)

# Here we fix the optimizer by using tf.train.AdamOptimizer
class Optimizer(object):
    def __init__(self, learning_rate, beta1, beta2, epsilon, name):
        # Parameters for optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = name

        # Container to store the gradients
        self._dev_opt = OrderedDict()
        self._dev_grad = OrderedDict()
        self._dev_grad_sum = OrderedDict()

    def collect_gradients(self, loss, vars):
        device = loss.device

        assert isinstance(vars, list) and len(vars) >= 1

        with tf.name_scope(self.name + "collect_grad"), tf.device(device):
            if device not in self._dev_opt:
                self._dev_opt[device] = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                               beta1=self.beta1,
                                                               beta2=self.beta2,
                                                               epsilon=self.epsilon,
                                                               name=self.name)
            grads = self._dev_opt[device].compute_gradients(loss, vars, gate_gradients=tf.train.Optimizer.GATE_NONE)
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]
            self._dev_grad[device] = grads

    def apply_update(self):
        device_list = list(self._dev_opt.keys())
        for dev in device_list:
            self._dev_grad_sum[dev] = []

        ops = []
        if len(device_list) > 1:
            with tf.name_scope("all_reduce"), tf.device(None):
                var_length = len(self._dev_grad[device_list[0]])
                for var_idx in range(var_length):
                    g = [self._dev_grad[dev][var_idx][0] for dev in device_list]
                    g = nccl_ops.all_sum(g)
                    for dev, gg in zip(device_list, g):
                        self._dev_grad_sum[dev].append((gg, self._dev_grad[dev][var_idx][1]))

            for dev_idx, (device, grads) in enumerate(self._dev_grad_sum.items()):
                with tf.name_scope("Apply_grad%d"%dev_idx), tf.device(device):
                    update_op = self._dev_opt[device].apply_gradients(grads)
                    ops.append(update_op)
        else:
            for device, grads in self._dev_grad.items():
                with tf.name_scope("Apply_grad"), tf.device(device):
                    update_op = self._dev_opt[device].apply_gradients(grads)
                    ops.append(update_op)

        ops.extend(self.reset_opt_state())
        return tf.group(*ops, name='TrainingOp')

    def reset_opt_state(self):
        local_ops = []
        for opt in self._dev_opt.values():
            local_ops.append(tf.variables_initializer(opt.variables()))
        return local_ops
