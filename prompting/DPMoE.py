import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import itertools
import wandb
import torch.distributed as dist

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
# from prompting.lasp import PromptLearner
from collections import OrderedDict

from .losses import domain_loss, class_loss
import random
import collections

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(prompts.shape, tokenized_prompts.shape)
        """
        # 假设 prompts 和 tokenized_prompts 是输入张量
        batch_size = 100
        num_batches = (prompts.size(0) + batch_size - 1) // batch_size

        # 初始化一个空列表来存储所有批次的结果
        all_results = []

        for i in range(num_batches):
            # print(i)
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, prompts.size(0))

            # 提取当前批次的数据
            prompts_batch = prompts[start_idx:end_idx]
            tokenized_prompts_batch = tokenized_prompts[start_idx:end_idx]

            # 处理当前批次的数据
            x = prompts_batch + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # 从 eot embedding 中提取 features（eot_token 是每个序列中的最大值）
            eot_indices = tokenized_prompts_batch.argmax(dim=-1)
            batch_result = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

            # 将当前批次的结果添加到 all_results 中
            all_results.append(batch_result)

            # 将所有批次的结果合并成一个 tensor
        final_result = torch.cat(all_results, dim=0)
        return final_result
        """
        # """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
        # """


def increase_top2_logits(logits, increment=1.0):
    """
    直接在 top-2 logits 上加上一个常数。

    参数:
        logits (torch.Tensor): 输入的 logits 向量。
        increment (float): 增加的值。

    返回:
        torch.Tensor: 修改后的 logits。
    """
    _, top2_indices = torch.topk(logits, k=2)  # 获取最高的两个logits的索引
    # print(top2_indices)
    increment_tensor = torch.zeros_like(logits)
    increment_tensor.scatter_(1, top2_indices, increment)

    logits += increment_tensor
    return logits


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        # cumulative_sum = [0] + list(itertools.accumulate(cfg.TRAINER.DPMoE.GROUPS))[:-1]
        cumulative_sum = len(cfg.DATASET.SOURCE_DOMAINS)
        self.num_domains = cumulative_sum
        ctx_init = cfg.TRAINER.DPMoE.CTX_INIT
        n_ctx = cfg.TRAINER.DPMoE.N_CTX
        self.class_token_position = []
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg

        self.n_ctx_before = []
        self.n_ctx_after = []
        prompt_template_before_all = [prompt.replace('.', '').split('{}')[0][:-1] for prompt in
                                      cfg.TRAINER.DPMoE.LASP_PROMPTS]
        prompt_template_after_all = [prompt.replace('.', '').split('{}')[1] for prompt in cfg.TRAINER.DPMoE.LASP_PROMPTS]
        self.num_domains = len(prompt_template_before_all)
        if ctx_init:
            # use given words to initialize context vectors
            for idx in range(self.num_domains):
                if len(prompt_template_before_all[idx]) == 0:
                    n_ctx = 0
                else:
                    n_ctx = len(prompt_template_before_all[idx].split(" "))
                prompt_template_before = prompt_template_before_all[idx].replace("_", " ")
                prompt = clip.tokenize(prompt_template_before)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors_before = embedding[0, 1: 1 + n_ctx, :]
                # Register expert 1 (soft prompt) with gradient
                self.register_parameter(f'ctx_before_soft_{idx}', nn.Parameter(ctx_vectors_before))
                # Register expert 2 (hard prompt) without gradient
                # self.register_buffer(f'ctx_before_hard_{idx}', ctx_vectors_before.clone())
                self.n_ctx_before.append(n_ctx)

                if len(prompt_template_after_all[idx]) == 0:
                    n_ctx = 0
                else:
                    n_ctx = len(prompt_template_after_all[idx].split(" "))
                prompt_template_after = prompt_template_after_all[idx].replace("_", " ")
                prompt = clip.tokenize(prompt_template_after)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors_after = embedding[0, 1: 1 + n_ctx, :]
                # Register expert 1 (soft prompt) with gradient
                self.register_parameter(f'ctx_after_soft_{idx}', nn.Parameter(ctx_vectors_after))
                # Register expert 2 (hard prompt) without gradient
                # self.register_buffer(f'ctx_after_hard_{idx}', ctx_vectors_after.clone())
                self.n_ctx_after.append(n_ctx)
        else:
            # random initialize
            for idx in range(self.num_domains):
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.register_parameter(f'ctx_before_soft_{idx}', nn.Parameter(ctx_vectors))
                self.n_ctx_before.append(n_ctx)

                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.register_parameter(f'ctx_after_soft_{idx}', nn.Parameter(ctx_vectors))
                self.n_ctx_after.append(n_ctx)
                prompt_prefix = " ".join(["X"] * n_ctx)
                prompt_suffix = " ".join(["X"] * n_ctx)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        self.tokenized_prompts_all_domains = []
        if ctx_init:
            for i in range(self.num_domains):
                prompts = [prompt_template_before_all[i] + ' ' + name + prompt_template_after_all[i] for
                       name in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
                self.tokenized_prompts_all_domains.append(tokenized_prompts)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                # These token vectors will be saved when in save_model(),
                # but they should be ignored in load_model() as we want to use
                # those computed using the current class names
                self.register_buffer(f"token_prefix_{i}", embedding[:, :1, :])  # SOS
                self.register_buffer(f"token_suffix_{i}", embedding[:, 1 + self.n_ctx_before[i]:, :])  # CLS, EOS
        else:
            for i in range(self.num_domains):
                prompts = [prompt_prefix + " " + name + " " + prompt_suffix + "." for name in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
                self.tokenized_prompts_all_domains.append(tokenized_prompts)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                # These token vectors will be saved when in save_model(),
                # but they should be ignored in load_model() as we want to use
                # those computed using the current class names
                self.register_buffer(f"token_prefix_{i}", embedding[:, :1, :])  # SOS
                self.register_buffer(f"token_suffix_{i}", embedding[:, 1 + self.n_ctx_before[i]:, :])  # CLS, EOS


        # self.gate = nn.Linear(vis_dim, self.num_domains)
        self.gate = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 2)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 2, vis_dim // 4)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(vis_dim // 4, self.num_domains))
        ]))
        # self.noise_linear = nn.Linear(vis_dim, self.num_domains)

        self.weights = nn.Parameter(torch.randn(self.num_domains))
        # self.weights = cfg.TRAINER.DPMoE.SH_WEIGHT

        self.n_cls = n_cls
        self.name_lens = name_lens
        self.classnames = classnames


    def construct_prompts(self, n_ctx_after, ctx_before, ctx_after, prefix, suffix, name_lens, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = []
        for i in range(len(prefix)):
            name_len = name_lens[i]
            prefix_i = prefix[i: i + 1, :, :]
            class_i = suffix[i: i + 1, :name_len, :]
            suffix_i = suffix[i: i + 1, name_len + n_ctx_after:, :]
            prompt = torch.cat(
                [
                    prefix_i,
                    ctx_before[i].unsqueeze(0),
                    class_i,
                    ctx_after[i].unsqueeze(0),
                    suffix_i,
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts



    def forward(self, all=False):
        prompts = []
        for i in range(self.num_domains):

            ctx_before_soft = getattr(self, f'ctx_before_soft_{i}')
            ctx_after_soft = getattr(self, f'ctx_after_soft_{i}')
            # ctx_before_hard = getattr(self, f'ctx_before_hard_{i}')
            # ctx_after_hard = getattr(self, f'ctx_after_hard_{i}')


            prefix = getattr(self, f"token_prefix_{i}")
            suffix = getattr(self, f"token_suffix_{i}")
            n_cls = self.n_cls
            name_lens = self.name_lens

            alpha = self.weights[i]
            # ctx_before = alpha * ctx_before_soft + (1 - alpha) * ctx_before_hard
            # ctx_after = alpha * ctx_after_soft + (1 - alpha) * ctx_after_hard
            ctx_before = ctx_before_soft
            ctx_after = ctx_after_soft


            ctx_before_i = ctx_before.unsqueeze(0).expand(n_cls, -1, -1)
            ctx_after_i = ctx_after.unsqueeze(0).expand(n_cls, -1, -1)


            pts_i = self.construct_prompts( self.n_ctx_after[i], ctx_before_i, ctx_after_i, prefix, suffix, name_lens )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.device = device
        # self.text_encoder = TextEncoder(clip_model)
        self.text_encoder = nn.DataParallel(TextEncoder(clip_model))
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model).to(
            clip_model.dtype)
        self.tokenized_prompts_all_domains = self.prompt_learner.tokenized_prompts_all_domains
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.num_domains = self.prompt_learner.num_domains
        # self.loss = contrastive_loss
        self.loss_class = class_loss
        self.loss_DA = domain_loss
        self.gate_count = collections.defaultdict(int)
        self.eps = 0.1
        self.max_iter = 100
        self.dataset = cfg.DATASET.NAME

    def compute_gated_prompts(self, image_features, tokenized_prompts_all, text_gating=None):
        domain_text_features_all = []
        if text_gating is None:
            gating_weights = self.prompt_learner.gate(image_features)
        else:
            gating_weights = text_gating

        gating_weights, indices = torch.topk(gating_weights, k=(len(tokenized_prompts_all)))

        prompts = self.prompt_learner()
        for i, prompt in enumerate(prompts):
            tokenized_prompts = tokenized_prompts_all[i]
            text_features_per_domain = self.text_encoder(prompt, tokenized_prompts)
            text_features_per_domain = text_features_per_domain / text_features_per_domain.norm(dim=-1,keepdim=True)
            domain_text_features_all.append(text_features_per_domain)
        text_features_all = []
        for batch_idx in range(len(image_features)):
            text_features = []
            for i in range(len(indices[batch_idx])):
                text_features.append(domain_text_features_all[indices[batch_idx][i]])
            text_features = torch.stack(text_features)
            text_features_all.append(text_features)

        gating_distribution = torch.softmax(gating_weights, dim=1)
        text_features = torch.einsum("bk,bkcd->bcd", gating_distribution, torch.stack(text_features_all))
        return text_features, domain_text_features_all

    def forward(self, image, label=None, domains=None):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts_all_domains = self.tokenized_prompts_all_domains

        text_features, domain_text_features_all = self.compute_gated_prompts(image_features, tokenized_prompts_all_domains)

        loss, logits = self.loss_class(image_features, text_features, label, t=self.logit_scale,)

        if self.prompt_learner.training:
            if self.cfg.TRAINER.DPMoE.ENABLE_CORRECTION:
                loss_DA = self.loss_DA(image_features, domain_text_features_all, label, domains,t=self.logit_scale)
                loss += self.cfg.TRAINER.DPMoE.DA_WEIGHT * loss_DA

        if self.prompt_learner.training:
            return loss
        else:
            return logits



@TRAINER_REGISTRY.register()
class DPMoE(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPMoE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DPMoE.PREC == "fp32" or cfg.TRAINER.DPMoE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        if cfg.TRAINER.DPMoE.FINETUNE_VIT_LN:
            print('Re-enabling LN...')
            for name, param in self.model.named_parameters():
                if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                    param.requires_grad_(True)

                    # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        # rank = dist.get_rank()
        self.model.to(self.device)
        # print(rank)
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.DPMoE.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        domain = domain.tolist()
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.DPMoE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, domain)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, domain)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        if isinstance(input, list):
            input = [inp.to(self.device, non_blocking=True) for inp in input]
        else:
            input = input.to(self.device, non_blocking=True)
        label = label.to(self.device)

        if self.cfg.DATALOADER.K_TRANSFORMS > 1:
            input = torch.cat(input)
            label = label.repeat(self.cfg.DATALOADER.K_TRANSFORMS)
        return input, label, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                print('Model not found at "{}", retrying to find one automatically...'.format(model_path))
                model_path = glob(f'{directory}/{name}/model-best.pth.tar-*')[0]
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            # checkpoint = load_checkpoint(model_path)
            checkpoint = torch.load(model_path, weights_only=False)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            ignore_list = ['ctx_after_hard','ctx_before_hard','token_prefix', 'token_suffix', 'token_prefix_all', 'token_suffix_all',
                           'class_text_features']
            ignore_list += [f'prompt_learner.{il}' for il in ignore_list]

            for k in ignore_list:
                state_dict.pop(k, None)
            for key in list(state_dict.keys()):
                if "token_prefix" in key:
                    del state_dict[key]
                if "token_suffix" in key:
                    del state_dict[key]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            w_weights = None
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in self._models[name].state_dict():
                    # if k == 'w':
                    #     w_weights = v
                    if v.size() == self._models[name].state_dict()[k].size():
                        new_state_dict[k] = v
                    else:
                        print(k, v.shape, self._models[name].state_dict()[k].size())
            print(f'Num of preserved keys: {len(new_state_dict)}')
            print(f'Keys: {new_state_dict.keys()}')
            # new_state_dict = {}
            self._models[name].load_state_dict(new_state_dict, strict=False)
        return w_weights

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
            # if dist.get_rank() == 0:
            # wandb.run.summary[tag] = v
            print(tag, v, file=open('result.txt', 'a'))

        with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        return list(results.values())[0]

