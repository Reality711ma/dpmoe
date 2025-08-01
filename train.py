import argparse
import os

import torch
import wandb
import torch.distributed as dist

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import prompting.DPMoE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.DA_W:
        cfg.TRAINER.DPMoE.DA_WEIGHT = args.DA_W

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.exp:
        cfg.EXP = args.exp


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.INCLUDE_ALL_CLASSES = False


    cfg.TRAINER.DPMoE = CN()

    cfg.TRAINER.DPMoE.DA_WEIGHT = 0.3 # weighf of LARS text-to-text loss
    cfg.TRAINER.DPMoE.N_CTX = 12  # number of context vectors
    cfg.TRAINER.DPMoE.CTX_INIT = False  # initialization words
    cfg.TRAINER.DPMoE.PREC = "amp"  # fp16, fp32, amp

    cfg.TRAINER.DPMoE.ENABLE_CORRECTION = False
    cfg.TRAINER.DPMoE.ENABLE_IMPLICIT_OP = 'sum' # mul
    cfg.TRAINER.DPMoE.PRETRAINED_PROMPTS_DIR = None
    cfg.TRAINER.DPMoE.TRAIN_W = True
    cfg.TRAINER.DPMoE.FINETUNE_VIT_LN = True




def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.TRAINER.temp = True
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    print(args.exp, file=open("result.txt", 'a'))
    print(cfg.DATASET.NAME, file=open('result.txt', 'a'))
    print(cfg.DATASET.SUBSAMPLE_CLASSES, file=open('result.txt', 'a'))
    print(args.exp, file=open('result.txt', 'a'))

    return cfg


def main(args):
    cfg = setup_cfg(args)
    # wandb.init(project="pl", entity="", name=args.exp, group=cfg.DATASET.NAME)
    # wandb.config.update(cfg)
    # wandb.save('train.py')
    # wandb.save('prompting/losses.py')
    # wandb.save('prompting/moe.py')
    # wandb.save(args.config_file)
    # if cfg.SEED >= 0:
    print("Setting fixed seed: {}".format(cfg.SEED))
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    # trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    # trainer.load_model(None, 10)

    if not args.no_train:
        trainer.train()


    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--DA_W", type=float, default=1.7, help="path to dataset")
    parser.add_argument("--SH_W", type=float, default=0.3, help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--exp", type=str, default="", help="experiment")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
