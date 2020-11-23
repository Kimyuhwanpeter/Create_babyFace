# -*- coding:utf-8 -*-
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

def residual_block(x, weight_decay):

    h = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)

    return tf.keras.layers.ReLU()(h + x)

def parents_generator(input_shape=(256, 256, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 256 x 256 x 64

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 128 x 128 x 128

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 64 x 64 x 256

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 32 x 32 x 256

    for i in range(5):
        h = residual_block(h, weight_decay) # 32 x 32 x 256

    h = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 64 x 64 x 256

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 128 x 128 x 128

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 256 x 256 x 64

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def baby_generator(input_shape=(256, 256, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 256 x 256 x 64

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 128 x 128 x 128

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 64 x 64 x 256

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 32 x 32 x 256

    for i in range(9):
        h = residual_block(h, weight_decay)

    h = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 64 x 64 x 256

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 128 x 128 x 128

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)   # 256 x 256 x 64

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discrim(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      weight_decay=0.00001,
                      norm='instance_norm'):
    regul = tf.keras.regularizers.l2
    dim_ = dim
    Norm = InstanceNormalization(epsilon=1e-5)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    Conv1 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', kernel_regularizer=regul(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv1)

    dim = min(dim * 2, dim_ * 8)
    Conv2 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv2)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    dim = min(dim * 2, dim_ * 8)
    Conv3 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv3)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    Conv4 = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv4)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)