import numpy as np
import argparse, os
import torch
from trainer import Trainer

torch.manual_seed(1)
np.random.seed(1)


def parse_args():
    """parsing and configuration"""
    desc = "Program Entry for Semantic Segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--test', type=str, default=False)
    parser.add_argument('--model', type=str, default='fcn16s')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory name to data location')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--pretrain', type=str, default='')

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    trainer = Trainer(args)
    if not args.test:
        trainer.train()
    else:
        if args.pretrain == '':
            print("Trained model unspecified")
            exit(-1)
        res = trainer.evaluate()
        print('Evaluation: Iou_mean: %.4f, Acc: %.4f,  ' % (
            res['iou_mean'], res['acc']))
    # trainer.generate_output()


if __name__ == '__main__':
    main()
