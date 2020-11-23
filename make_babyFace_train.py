# -*- coding:utf-8 -*-
from absl import flags
from make_babyFace_model import *
from random import random, shuffle

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

flags.DEFINE_string("A_txt_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[1]Father/labels.txt", "Training A text path")

flags.DEFINE_string("A_img_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[1]Father/AFAD/", "Training A image path")

flags.DEFINE_string("B_txt_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[2]Mother/labels.txt", "Training B text path")

flags.DEFINE_string("B_img_path", "D:/[1]DB/[2]third_paper_DB/[2]Parent_face/[2]Mother/AFAD/", "Training B image path")

flags.DEFINE_string("C_txt_path", "D:/[1]DB/[2]third_paper_DB/[1]UTK_baby/images/labels.txt", "Training C text path")

flags.DEFINE_string("C_img_path", "D:/[1]DB/[2]third_paper_DB/[1]UTK_baby/images/", "Training C image path")

flags.DEFINE_integer("load_size", 266, "Load size before input the model")

flags.DEFINE_integer("img_size", 256, "Model input size")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Training epochs")

flags.DEFINE_integer("learning_decay", 100, "Learning rate decay stap")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint")

flags.DEFINE_string("save_samples", "", "Save training sample images")

flags.DEFINE_string("graphs", "", "Save loss graphs")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

len_dataset = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)
len_dataset = len(len_dataset)
G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.learning_decay * len_dataset)
D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.learning_decay * len_dataset)
g_optim = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

def tr_C_input_func(C):

    C_img = tf.io.read_file(C)
    C_img = tf.image.decode_jpeg(C_img, 3)
    C_img = tf.image.resize(C_img, [FLAGS.load_size, FLAGS.load_size])
    C_img = tf.image.random_crop(C_img, [FLAGS.img_size, FLAGS.img_size, 3])

    if random() > 0.5:
        C_img = tf.image.flip_left_right(C_img)

    C_img = C_img / 127.5 - 1.

    return C_img

def tr_input_func(A, B):

    A_img, B_img = tf.io.read_file(A), tf.io.read_file(B)
    A_img = tf.image.decode_jpeg(A_img, 3)
    B_img = tf.image.decode_jpeg(B_img, 3)

    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3])

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    A_img = A_img / 127.5 - 1.
    B_img = B_img / 127.5 - 1.

    return A_img, B_img

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(from_father_mod, from_mother_mod, from_baby_mod,
            father_discrim, mother_discrim, baby_discrim,
            A_images, B_images, C_images):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        father_part = run_model(from_father_mod, A_images, True)
        mother_part = run_model(from_mother_mod, B_images, True)
        similarity = (father_part * mother_part) / (tf.math.sqrt(father_part*father_part) * tf.math.sqrt(mother_part*mother_part))

        baby_part = run_model(from_baby_mod, similarity, True)
        a = 0

    return g_loss, d_loss

def main():
    from_father_mod = parents_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    from_mother_mod = parents_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    from_baby_mod = baby_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    father_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    mother_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    baby_discrim = discrim(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    from_father_mod.summary()
    from_mother_mod.summary()
    from_baby_mod.summary()
    father_discrim.summary()
    mother_discrim.summary()
    baby_discrim.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(from_father_mod=from_father_mod,
                                   from_mother_mod=from_mother_mod,
                                   from_baby_mod=from_baby_mod,
                                   g_optim=g_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("====================================================================")
            print("Succeed restoring '{}'".format(ckpt_manager.latest_checkpoint))
            print("====================================================================")

    if FLAGS.train:
        count = 0

        A_img = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_img = [FLAGS.A_img_path + img for img in A_img]

        B_img = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_img = [FLAGS.B_img_path + img for img in B_img]

        C_img = np.loadtxt(FLAGS.C_txt_path, dtype="<U100", skiprows=0, usecols=0)
        C_img = [FLAGS.C_img_path + img for img in C_img]

        for epoch in range(FLAGS.epochs):

            AB_gener = tf.data.Dataset.from_tensor_slices((A_img, B_img))
            AB_gener = AB_gener.shuffle(len(A_img))
            AB_gener = AB_gener.map(tr_input_func)
            AB_gener = AB_gener.batch(FLAGS.batch_size)
            AB_gener = AB_gener.prefetch(tf.data.experimental.AUTOTUNE)

            C_gener = tf.data.Dataset.from_tensor_slices(C_img)
            C_gener = C_gener.shuffle(len(C_img))
            C_gener = C_gener.map(tr_C_input_func)
            C_gener = C_gener.batch(FLAGS.batch_size + 1)
            C_gener = C_gener.prefetch(tf.data.experimental.AUTOTUNE)

            AB_iter = iter(AB_gener)
            AB_idx = len(A_img) // FLAGS.batch_size
            for step in range(AB_idx):
                A_images, B_images = next(AB_iter)
                C_iter = iter(C_gener)
                C_images = next(C_iter)

                G_loss, D_loss = cal_loss(from_father_mod, from_mother_mod, from_baby_mod,
                         father_discrim, mother_discrim, baby_discrim,
                         A_images, B_images, C_images)

if __name__ == "__main__":
    main()