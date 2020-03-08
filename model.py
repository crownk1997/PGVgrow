import tensorflow as tf
import numpy as np
from tqdm import tqdm
import imageio
import time
from tfuilts import *
from networks import *
from optimizer import *


class PGVgrow(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.outpath = args.outpath
        self.divergence = args.divergence
        self.step_size = args.step_size
        self.z_dim = args.z_dim
        self.num_gpus = args.num_gpu
        self.use_gp = args.use_gp
        self.target_gp = args.target_gp
        self.coef_gp = args.coef_gp
        self.dur_nimg = args.dur_nimg
        self.total_nimg = args.total_nimg
        self.init_res = args.init_res
        self.U = args.U
        self.T = args.T
        self.L = args.L
        self.num_row = args.num_row
        self.num_col = args.num_col
        self.coef_smooth = args.coef_smooth
        self.pool_size = args.pool_size
        self.resume_training = args.resume_training
        self.resume_num = args.resume_num
        self.batchsize_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}

        print("Start building graph...")
        self.build_model()

    def inference_dataset(self):
        tfrecord_dir = os.path.join("datasets", self.dataset)
        self.num_channels, self.resolution, self.num_features = inferenceResolution(tfrecord_dir)

    def build_model(self):
        print("Inference dataset information...")
        # inference the input dataset and set the related parameters
        self.inference_dataset()

        with tf.name_scope('Inputs'):
            self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.resolution, self.resolution, self.num_channels], name='images')
            self.latent_minibatch = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='batch_latents')
            self.latent_pool = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='pool_latents')
            self.particle_minibatch = tf.placeholder(dtype=tf.float32, shape=[None, self.resolution, self.resolution, self.num_channels],
                                   name='batch_particles')
            self.particle_pool = tf.placeholder(dtype=tf.float32, shape=[None, self.resolution, self.resolution, self.num_channels],
                                    name='pool_particles')
            self.lod_in = tf.placeholder(dtype=tf.float32, shape=[], name='level_of_details')
            self.mix_factors = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 1], name='mix_factors')

            # split input according to number of gpu
            self.images_pergpu = tf.split(self.images, self.num_gpus)
            self.latent_minibatch_pergpu = tf.split(self.latent_minibatch, self.num_gpus)
            self.latent_pool_pergpu = tf.split(self.latent_pool, self.num_gpus)
            self.particle_minibatch_pergpu = tf.split(self.particle_minibatch, self.num_gpus)
            self.particle_pool_pergpu = tf.split(self.particle_pool, self.num_gpus)
            self.mix_factors_pergpu = tf.split(self.mix_factors, self.num_gpus)

        # Build optimizer for generator and discriminator
        self.opt_G = Optimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_G')
        self.opt_D = Optimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_D')

        self.pool_images = []
        self.pool_grads = []
        self.pool_d_score = []
        self.pool_smoo = []
        for gpu in range(self.num_gpus):
            with tf.device("/GPU:%d"%gpu):
                if gpu == 0:
                    self.G = generator(self.latent_minibatch_pergpu[gpu], self.lod_in, self.num_channels, self.resolution,
                                       self.z_dim, self.num_features)
                    self.G_smooth = generator(self.latent_minibatch_pergpu[gpu], self.lod_in, self.num_channels,
                                              self.resolution, self.z_dim, self.num_features, is_smoothing=True)
                    self.d_real_logits = discriminator(self.images_pergpu[gpu], self.lod_in, self.num_channels, self.resolution, self.num_features)
                else:
                    self.G = generator(self.latent_minibatch_pergpu[gpu], self.lod_in, self.num_channels, self.resolution,
                                       self.z_dim, self.num_features, reuse=True)
                    self.d_real_logits = discriminator(self.images_pergpu[gpu], self.lod_in, self.num_channels, self.resolution,
                                                       self.num_features, reuse=True)

                self.d_fake_logits = discriminator(self.particle_minibatch_pergpu[gpu], self.lod_in, self.num_channels,
                                                   self.resolution, self.num_features, reuse=True)

                self.pool_images.append(self.G)
                self.pool_d_score.append(self.d_fake_logits)
                self.pool_smoo.append(self.G_smooth)
                # variables G and variables D:
                vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
                vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

                with tf.name_scope("D_loss%d"%gpu):
                    self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logits,
                                                                                         labels=tf.ones_like(
                                                                                             self.d_real_logits)))
                    self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logits,
                                                                                         labels=tf.zeros_like(
                                                                                             self.d_fake_logits)))
                    self.loss_d = self.loss_d_real + self.loss_d_fake

                    if self.use_gp:
                        with tf.name_scope("GP%d"%gpu):
                            mix_images = self.images_pergpu[gpu] + self.mix_factors_pergpu[gpu] * (self.particle_minibatch_pergpu[gpu] - self.images_pergpu[gpu])
                            d_mix_logits = discriminator(mix_images, self.lod_in, self.num_channels, self.resolution,
                                                         self.num_features, reuse=True)
                            d_mix_grad = tf.gradients(d_mix_logits, mix_images)[0]
                            mix_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(d_mix_grad), axis=[1, 2, 3]))
                            mix_grad_penalty = tf.square(mix_grad_norm - self.target_gp)
                            self.loss_d += 0.5 * self.coef_gp * tf.reduce_mean(mix_grad_penalty)

                # computing gradient:
                self.d_fake_grad = tf.gradients(self.d_fake_logits, self.particle_minibatch_pergpu[gpu])[0]

                self.pool_grads.append(self.d_fake_grad)

                with tf.name_scope("G_loss%d"%gpu):
                    self.loss_g = 0.5 * tf.reduce_mean(tf.reduce_sum((self.G - self.particle_pool_pergpu[gpu]) ** 2))

                self.opt_G.collect_gradients(self.loss_g, vars_g)
                self.opt_D.collect_gradients(self.loss_d, vars_d)

        self.update_D = self.opt_D.apply_update()
        self.update_G = self.opt_G.apply_update()

        self.reset_D = tf.group(*self.opt_D.reset_opt_state(), name="reset_D")
        self.reset_G = tf.group(*self.opt_G.reset_opt_state(), name="reset_G")

        # ========== Moving average of generator ==========
        G_trainables = OrderedDict(
            [(var.name[len('generator/'):], var) for var in tf.trainable_variables('generator' + '/')])
        Gs_trainables = OrderedDict([(var.name[len('generator_smoothing/'):], var) for var in
                                     tf.trainable_variables('generator_smoothing' + '/')])

        with tf.name_scope('generator_smoothing/'):
            with tf.name_scope('smoothing'):
                ops = []
                for name, var in Gs_trainables.items():
                    new_value = G_trainables[name] + (var - G_trainables[name]) * self.coef_smooth
                    ops.append(var.assign(new_value))
                self.update_gs = tf.group(*ops)

    def train(self):

        saver = tf.train.Saver()

        resolution_log2 = int(np.log2(self.resolution))

        out_path = os.path.join(self.outpath, '%s-%s' % (self.dataset, self.divergence))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with tf.Session() as sess:
            iterators = [data_tool.data_iterator(dataset=self.dataset, lod_in=lod,
                                                 batch_size=self.batchsize_dict[2 ** (resolution_log2 - lod)] * self.num_gpus,
                                                 resolution_log2=resolution_log2) for lod in
                         range(int(np.log2(self.resolution / self.init_res)) + 1)]

            if not self.resume_training:
                sess.run(tf.global_variables_initializer())
                num_img = 0
                tick_kimg = 0
                prev_lod = -1.0

            else:
                saver.restore(sess, os.path.join(out_path, 'networks-%08d.ckpt' % self.resume_num))
                num_img = self.resume_num
                tick_kimg = (num_img // 1000)

            cur_lod = self.lod(num_img)
            z_dim = ((self.num_row * self.num_col) // self.num_gpus + 1) * self.num_gpus * self.num_gpus
            z_fixed = np.random.randn(z_dim, self.z_dim)
            count = 0

            training_start_time = time.time()
            while num_img <= self.total_nimg:
                prev_lod = cur_lod

                # get mini-batch size:
                batch_size = self.batchsize_dict[2 ** int(resolution_log2 - np.floor(cur_lod))] * self.num_gpus

                # sample a latent pool and get particles:
                Sz = np.random.randn(batch_size * self.pool_size, self.z_dim)
                P = sess.run(self.pool_images, feed_dict={self.latent_minibatch: Sz, self.lod_in: cur_lod})
                P = np.concatenate(P, axis=0)
                # inner loop:
                for t in range(self.T):

                    # optimize discriminator:
                    for u in range(self.U):

                        # get a batch of real images:
                        x = next(iterators[int(np.floor(cur_lod))])
                        x = process_real(x, cur_lod)
                        num_img += batch_size

                        # sample a batch of latents from the pool:
                        sample_index = np.random.choice(batch_size * self.pool_size, batch_size, replace=False)

                        # update
                        if self.use_gp:
                            mix_coef = np.random.uniform(0, 1, [batch_size, 1, 1, 1])
                            sess.run([self.update_D, self.update_gs],
                                     feed_dict={self.images: x, self.particle_minibatch: P[sample_index],
                                                self.mix_factors: mix_coef, self.lod_in: cur_lod})
                        else:
                            sess.run([self.update_D, self.update_gs], feed_dict={self.images: x,
                                                                                 self.particle_minibatch: P[sample_index],
                                                                                 self.lod_in: cur_lod})

                    # move particles
                    d_score = sess.run(self.pool_d_score, feed_dict={self.particle_minibatch: P, self.lod_in: cur_lod})
                    d_score = np.concatenate(d_score, axis=0)
                    grad = sess.run(self.pool_grads, feed_dict={self.particle_minibatch: P, self.lod_in: cur_lod})
                    grad = np.concatenate(grad, axis=0)
                    P += self.coef_div(d_score) * grad

                    # optimize generator:
                    for l in range(self.L):
                        sess.run(self.update_G, feed_dict={self.latent_minibatch: Sz, self.particle_pool: P,
                                                           self.lod_in: cur_lod})

                    cur_lod = self.lod(num_img)

                    # reset Adam optimizers states when increasing resolution:
                    if np.floor(cur_lod) != np.floor(prev_lod) or np.ceil(cur_lod) != np.ceil(prev_lod):
                        sess.run([self.reset_D, self.reset_G])

                    if (num_img // 1000) >= tick_kimg + 150:
                        count += 1

                        tick_kimg = (num_img // 1000)
                        real_loss, fake_loss = sess.run([self.loss_d_real, self.loss_d_fake],
                                                        feed_dict={self.images: x, self.particle_minibatch: P[sample_index],
                                                                   self.lod_in: cur_lod})
                        G_loss = sess.run(self.loss_g, feed_dict={self.latent_minibatch: Sz, self.particle_pool: P,
                                                                  self.lod_in: cur_lod})

                        cur_time = time.time()
                        print('num_img: %d ' % num_img, '  |  lod_in: %.2f' % cur_lod, '\n',
                              'D real loss: %.6f' % real_loss, '  |  D fake loss: %.6f' % fake_loss,
                              '  |  Projection loss: %.6f' % G_loss, ' | Training time: %-12s sec' % format_time(cur_time - training_start_time))

                        gen_imgs = []
                        for i in range(self.num_col * self.num_row):
                            img = sess.run(self.G_smooth, feed_dict={self.latent_minibatch: z_fixed[i:i+self.num_gpus], self.lod_in: cur_lod})
                            gen_imgs.append(img[0])
                        gen_imgs = np.array(gen_imgs)
                        gen_imgs = (gen_imgs + 1) / 2
                        imageio.imsave(os.path.join(out_path, 'fakes%06d.png' % (num_img // 1000)),
                                       montage(gen_imgs, grid=[self.num_row, self.num_col]))

    def coef_div(self, d_score):

        if self.divergence == 'KL':
            s = np.ones_like(d_score)
        elif self.divergence == 'LogD':
            s = 1 / (1 + np.exp(d_score))
        elif self.divergence == 'JS':  # ensure numerical stablity
            s = 1 / (1 + 1 / (1e-6 + np.exp(d_score)))
        elif self.divergence == 'Jef':  # ensure numerical stablity
            s = np.clip(1 + np.exp(d_score), 1, 100)

        return self.step_size * np.reshape(s, [-1, 1, 1, 1])

    def lod(self, num_img):

        ph_num = num_img // (2 * self.dur_nimg)
        remain_num = num_img - ph_num * (2 * self.dur_nimg)

        if np.log2(self.resolution / self.init_res) <= ph_num:
            return 0.
        elif remain_num <= self.dur_nimg:
            return np.log2(self.resolution / self.init_res) - ph_num
        else:
            return np.log2(self.resolution / self.init_res) - ph_num - \
                   (remain_num - self.dur_nimg) / self.dur_nimg



