import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset

def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    gt_dir = osp.join(root_dir, 'gt')
    pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split('+')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split('+')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset),
                                 osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda, cfg.all_metrics)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='CIR-SwinNet+CIRNet_R50')
    parser.add_argument('--datasets', type=str, default='DUT+LFSD+NJU2K+NLPR+SIP+STERE')
    parser.add_argument('--root_dir', type=str, default='D:\\data\\deepLearn\\code\\deep-learning-for-image-processing-master\\pytorch_classification\\CIRNet_TIP2022-main2\\eval')
    parser.add_argument('--save_dir', type=str, default='./Resultss')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--all_metrics', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    main(config)
