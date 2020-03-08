from model import PGVgrow
import argparse
import os
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Arguments for training PGVgrow')

# ========== Arguments about training model ==========
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU(s) to use to train model')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of gpus to use to train model')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Dataset to use to train model')
parser.add_argument('--divergence', type=str, default='KL',
                    help='Divergence to use in model (KL, JS, Logd, Jef)')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')
parser.add_argument('--init_res', type=int, default=4,
                    help='Initial resolution of images to train')
parser.add_argument('--z_dim', type=int, default=512,
                    help='Dimension of latent vectors')
parser.add_argument('--dur_nimg', type=int, default=600000,
                    help='Number of images used for a phase training')
parser.add_argument('--total_nimg', type=int, default=18000000,
                    help='Number of images used for the whole training')
parser.add_argument('--pool_size', type=int, default=1,
                    help='Number of minibatches in a pool')
parser.add_argument('--T', type=int, default=1,
                    help='Number of loops for moving particles')
parser.add_argument('--U', type=int, default=1,
                    help='Number of loops for training D')
parser.add_argument('--L', type=int, default=1,
                    help='Number of loops for training G')
parser.add_argument('--step_size', type=float, default=1.0,
                    help='Step size for moving particles')
parser.add_argument('--G_step', type=int, default=1,
                    help='Number of loops for training G')
parser.add_argument('--D_step', type=int, default=1,
                    help='Number of loops for training D')

parser.add_argument('--use_gp', type=bool, default=True,
                    help='Whether to use gradient penalty in model')
parser.add_argument('--coef_gp', type=float, default=1.0,
                    help='The coefficient of gradient penalty')
parser.add_argument('--target_gp', type=float, default=1.0,
                    help='The coefficient of target gradient penalty')

parser.add_argument('--coef_smooth', type=float, default=0.99,
                    help='The coefficient of generator moving average')

# ========== Arguments about output result ==========
parser.add_argument('--outpath', type=str, default='./results',
                    help='The output path for training result')
parser.add_argument('--num_row', type=int, default=10,
                    help='The number of images in a col of image grid')
parser.add_argument('--num_col', type=int, default=10,
                    help='The number of images in a row of image grid')

# ========== Arguments about resume training ==========
parser.add_argument('--resume_training', type=bool, default=False,
                    help='Whether to resume training')
parser.add_argument('--resume_num', type=int, default=0,
                    help='The resuming number of images')

args = parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    pgvgrow = PGVgrow(args)
    print("Start training...")
    pgvgrow.train()