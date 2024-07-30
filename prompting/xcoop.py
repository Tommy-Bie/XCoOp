import os.path as osp
from glob import glob
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .losses import contrastive_loss, contrastive_loss_token_level

import wandb
import numpy as np
from torch.nn import functional as F

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME  # ViT-B/16
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
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_encoder_model, all_classnames):
        super().__init__()
        n_cls = len(all_classnames) if cfg.DATASET.INCLUDE_ALL_CLASSES else len(classnames) 
        n_ctx = cfg.TRAINER.XCoOp.N_CTX  # 4: each learnable prompt consists of 4 words (initialize using "a photo of a" in CoOp)
        ctx_init = cfg.TRAINER.XCoOp.CTX_INIT  # initial context: "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # context dimension
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim  # visual embedding dimension, which is the output dimension of visual encoder
        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # 224, cfg.INPUT.SIZE: (224, 224)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")  # "a photo of a"
            n_ctx = len(ctx_init.split(" "))  
            prompt = clip.tokenize(ctx_init)  
            with torch.no_grad():             
                embedding = clip_model.token_embedding(prompt).type(dtype) 
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] 
            ctx_vectors_all = ctx_vectors.repeat(n_cls, 1).reshape(n_cls, n_ctx, ctx_dim) 
            prompt_prefix = ctx_init  # "a photo of a"
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")  # 4

        self.ctx = nn.Parameter(ctx_vectors_all)  # the embedding of the learnable prompts prefix

        classnames = [name.replace("_", " ") for name in classnames] 
        all_classnames = [name.replace("_", " ") for name in all_classnames] 

        if cfg.DATASET.INCLUDE_ALL_CLASSES:
            # Preserve class order
            classes_delta = [name for name in all_classnames if name not in classnames]
            print(f'Number of extra class names: {len(classes_delta)}')
            classnames += classes_delta
            print(f'Number of class names after: {len(classnames)}')
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if ctx_init:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]  # prompts: "a photo of a [classname]"
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn): tokenize the prompts
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # embedding of the prompts

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS: start of string
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS: cls token & end of string

        if cfg.TRAINER.XCoOp.ENABLE:
            self.construct_references_xcoop_clinical(cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx)

        self.n_cls = n_cls  
        self.n_ctx = n_ctx 
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens  
        self.all_classnames = all_classnames  

        if cfg.TRAINER.XCoOp.ENABLE_W:
            self.w = nn.Parameter(torch.zeros(1, ctx_dim, device=embedding.device, dtype=dtype), requires_grad=self.cfg.TRAINER.XCoOp.ENABLE_W)

    # NOTE: clinical prompts
    def construct_references_xcoop_clinical(self, cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx):
        print("Initializing Clinical prompts...")
        template_prompts = cfg.TRAINER.XCoOp.CLINICAL_PROMPTS
        all_classnames = [name.replace("_", " ") for name in all_classnames]
        print(f'Num of classes: {len(all_classnames)}')

        all_token_embeddings = []
        all_class_text_features = []
        prompts = []
        if cfg.DATASET.NAME == "Pneumonia":
            prompts = [
                template_prompts[0].format(all_classnames[0]),  # normal
                template_prompts[1].format(all_classnames[1]),  # pneumonia
                ]
        else:  # TODO: other datasets
            print("Please create the hand-crafted prompts when using new datasets")

        tokenized_prompts_all_c = torch.cat([clip.tokenize(p) for p in prompts])
        text_encoder_model.cuda()
        with torch.no_grad():
            embedding_all_cls = clip_model.token_embedding(tokenized_prompts_all_c).cuda().type(dtype)  # embedding of the hand-crafted prompts,
            class_text_features = text_encoder_model(embedding_all_cls, tokenized_prompts_all_c).type(dtype)  # text features of the hand-crafted prompts
            all_class_text_features.append(class_text_features)  
            all_token_embeddings.append(embedding_all_cls)
        
        self.register_buffer("class_token_embeddings", torch.stack(all_token_embeddings, dim=0))
        self.register_buffer("class_text_features", torch.stack(all_class_text_features, dim=0))

        prompts = [prompt_prefix + " " + name + "." for name in all_classnames]  
        tokenized_prompts_all_c_ = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_all_c_).type(dtype)

        self.register_buffer("token_prefix_all", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_all", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts_all = tokenized_prompts_all_c  # the clinical prompts
        self.tokenized_prompts_all_c_ = tokenized_prompts_all_c_  # the coop prompts: "a photo of a [cls]"
        self.n_cls_all = len(prompts)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, all=False):
        if not all:
            prefix = self.token_prefix
            suffix = self.token_suffix
            n_cls = self.n_cls
        else:
            prefix = self.token_prefix_all  # EOS: (n_cls, 1, ctx_dim)
            suffix = self.token_suffix_all  
            n_cls = len(self.all_classnames)
        ctx = self.ctx # (n_cls, n_ctx, ctx_dim)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
        
        return prompts


# TODO: **********************************************************************
# TODO: ******************** XCoOp Prompt Learner ***********************
# TODO: **********************************************************************
class XCoOpPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_encoder_model, all_classnames):
        super().__init__()
        n_cls = len(all_classnames) if cfg.DATASET.INCLUDE_ALL_CLASSES else len(classnames) 
        n_ctx = cfg.TRAINER.XCoOp.N_CTX  # 4: each learnable prompt consists of 4 words (initialize using "a photo of a" in CoOp)
        ctx_init = cfg.TRAINER.XCoOp.CTX_INIT  # initial context: "a photo of a"
        dtype = clip_model.dtype  # torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # context dimension
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim  # visual embedding dimension, which is the output dimension of visual encoder
        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # 224, cfg.INPUT.SIZE: (224, 224)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg
        self.N = cfg.TRAINER.XCoOp.TEXT_PROMPT_NUMBER 
        self.M = self.cfg.TRAINER.XCoOp.VISUAL_FEATURE_NUMBER

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")  # "a photo of a"
            n_ctx = len(ctx_init.split(" "))  
            prompt = clip.tokenize(ctx_init)  
            with torch.no_grad():            
                embedding = clip_model.token_embedding(prompt).type(dtype)  
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  # embedding of the prompts (ctx) 
            ctx_vectors_all = ctx_vectors.repeat(n_cls, 1).reshape(n_cls, n_ctx, ctx_dim) 
            prompt_prefix = ctx_init  # "a photo of a"

        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")  # 4
        
        ctx_vectors_all = ctx_vectors_all.repeat(self.N, 1, 1)  # (n_cls*N, n_ctx, ctx_dim)
        self.ctx = nn.Parameter(ctx_vectors_all)  # the embedding of the learnable prompts

        classnames = [name.replace("_", " ") for name in classnames] 
        all_classnames = [name.replace("_", " ") for name in all_classnames] 

        if cfg.DATASET.INCLUDE_ALL_CLASSES:
            # Preserve class order
            classes_delta = [name for name in all_classnames if name not in classnames]
            print(f'Number of extra class names: {len(classes_delta)}')
            classnames += classes_delta
            print(f'Number of class names after: {len(classnames)}')
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        if ctx_init:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]  # prompts: "a photo of a [cls]"
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # tokenize the prompts
            # NOTE: N textual prompts for each class
            tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # embedding of the prompts

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS: start of string 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS: cls token & end of string

        if cfg.TRAINER.XCoOp.ENABLE:
            self.construct_references_xcoop_clinical(cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx)

        self.n_cls = n_cls  
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        
        self.name_lens = name_lens 
        self.all_classnames = all_classnames 

        if cfg.TRAINER.XCoOp.ENABLE_W:
            self.w = nn.Parameter(torch.zeros(1, ctx_dim, device=embedding.device, dtype=dtype), requires_grad=self.cfg.TRAINER.XCoOp.ENABLE_W)  # (1, ctx_dim)

    # NOTE: clinical prompts
    def construct_references_xcoop_clinical(self, cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx):
        print("Initializing Clinical prompts...")
        template_prompts = cfg.TRAINER.XCoOp.CLINICAL_PROMPTS
        all_classnames = [name.replace("_", " ") for name in all_classnames]
        print(f'Num of classes: {len(all_classnames)}')

        all_token_embeddings = []
        all_class_text_features = []
        prompts = []
        if cfg.DATASET.NAME == "Pneumonia":
            prompts = [
                template_prompts[0].format(all_classnames[0]),  # normal
                template_prompts[1].format(all_classnames[1]),  # pneumonia
                ]
        else:  # NOTE: for other datasets
            print("Please create the hand-crafted prompts when using new datasets")
            
        tokenized_prompts_all_c = torch.cat([clip.tokenize(p) for p in prompts]) # (2, 77)

        tokenized_prompts_all_c = tokenized_prompts_all_c.repeat(self.N, 1)  
        text_encoder_model.cuda()
        with torch.no_grad():
            embedding_all_cls = clip_model.token_embedding(tokenized_prompts_all_c).cuda().type(dtype)  # embedding of the hand-crafted prompts
            class_text_features = text_encoder_model(embedding_all_cls, tokenized_prompts_all_c).type(dtype)  # text features of the hand-crafted prompts
            all_class_text_features.append(class_text_features)  
            all_token_embeddings.append(embedding_all_cls)
        
        self.register_buffer("class_token_embeddings", torch.stack(all_token_embeddings, dim=0))  
        self.register_buffer("class_text_features", torch.stack(all_class_text_features, dim=0))  

        prompts = [prompt_prefix + " " + name + "." for name in all_classnames]  # "a photo of a [cls]."
        tokenized_prompts_all_c_ = torch.cat([clip.tokenize(p) for p in prompts]) 
        tokenized_prompts_all_c_ = tokenized_prompts_all_c_.repeat(self.N, 1)  # (n_cls*N, n_tkn)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_all_c_).type(dtype) 
            
        self.register_buffer("token_prefix_all", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_all", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts_all = tokenized_prompts_all_c  # the clinical prompts
        self.tokenized_prompts_all_c_ = tokenized_prompts_all_c_  # the coop prompts: "a photo of a [cls]"
        self.n_cls_all = len(prompts)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)  
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, all=False):
        if not all:
            prefix = self.token_prefix
            suffix = self.token_suffix
            n_cls = self.n_cls
        else:
            prefix = self.token_prefix_all  # EOS: (n_cls, 1, ctx_dim)
            suffix = self.token_suffix_all  
            n_cls = len(self.all_classnames)
        ctx = self.ctx 
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (n_cls * N, n_tkn, ctx_dim), e.g., (8, 77, 512)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_classnames):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = nn.DataParallel(TextEncoder(clip_model))  # nn.DataParallel: multiple GPUs
        
        if cfg.TRAINER.XCoOp.TEXT_PROMPT_NUMBER > 1: 
            self.prompt_learner = XCoOpPromptLearner(cfg, classnames, clip_model, self.text_encoder, all_classnames)
        else: 
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model, self.text_encoder, all_classnames)
        
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # (N*n_cls, 77), tokenization of "a photo of a [cls]"
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.loss = contrastive_loss
        self.token_loss = contrastive_loss_token_level
        self.n_cls = len(all_classnames) if cfg.DATASET.INCLUDE_ALL_CLASSES else len(classnames)
        self.N = cfg.TRAINER.XCoOp.TEXT_PROMPT_NUMBER
        

    # NOTE: clinical concept guided optimization
    def forward_text_to_text_clinical(self):
        with torch.no_grad():
            class_text_features = self.prompt_learner.class_text_features  # clinical prompt text features
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        if torch.rand(1).item() < 0.5:
            noise = 0.05 * torch.randn_like(class_text_features)
            class_text_features.add_(noise)

        prompts = self.prompt_learner(all=True) # learnable/soft prompts
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_all_c_)  
        
        if prompts.shape[0] != self.n_cls: 
            text_features = text_features.contiguous().view(self.N, self.n_cls, -1)  
            text_features = text_features.mean(dim=0) 
            class_text_features = class_text_features.contiguous().view(self.N, self.n_cls, -1) 
            class_text_features = class_text_features.mean(dim=0, keepdim=True)

        if self.cfg.TRAINER.XCoOp.ENABLE_W:
            w = self.prompt_learner.w
            text_features = text_features + w

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0)

        label = torch.arange(self.prompt_learner.n_cls_all, device=class_text_features.device, dtype=torch.long).unsqueeze(0).expand(class_text_features.size(0), -1)
        
        # self.loss: contrastive_loss
        # text_features: text features of soft prompts
        # class_text_features: text featutures of clinical prompts
        # label: the class labels (expand to the shape the text features in order to calculate cross entropy loss)
        # t: just a scale parameter

        # return: contrastive loss returns two items (but here only return the loss):
        # 1. cross entropy of logits and labels
        # 2. logits: the multiplication (or similarity, implemented using @) of text_features and class_text_features
        return self.loss(text_features, class_text_features, label, t=self.logit_scale)[0]  
    
    # NOTE: token-level hard-soft prompt alignment
    def forward_token_to_token_clinical(self):
        """ hard-soft prompt token-level alignment: align the token embeddings """

        with torch.no_grad():
            class_token_embeddings = self.prompt_learner.class_token_embeddings 
            class_token_embeddings = class_token_embeddings / class_token_embeddings.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(all=True)  # the token embedding of the soft prompt
        prompts = prompts / prompts.norm(dim=-1, keepdim=True)
        prompts = prompts.unsqueeze(0)  

        if prompts.shape[1] > self.n_cls:
            class_token_embeddings = class_token_embeddings.contiguous().view(1, self.N, self.n_cls, class_token_embeddings.shape[-2], class_token_embeddings.shape[-1])
            class_token_embeddings = class_token_embeddings.mean(dim=1)  
            prompts = prompts.contiguous().view(1, self.N, self.n_cls, prompts.shape[-2], prompts.shape[-1])
            prompts = prompts.mean(dim=1)  

        label = torch.arange(self.prompt_learner.n_cls_all, device=class_token_embeddings.device, dtype=torch.long).unsqueeze(0).expand(class_token_embeddings.size(0), -1)
        return self.token_loss(prompts, class_token_embeddings, label, t=self.logit_scale)[0]  

    def forward_image_text(self, image, label=None):
        
        batch_size = image.shape[0]
        M = self.cfg.TRAINER.XCoOp.VISUAL_FEATURE_NUMBER
        image_features, image_features_local = self.image_encoder(image.type(self.dtype), M)  
        if M <= 1:
            image_features = image_features.unsqueeze(dim=0)
        image_features_global = image_features.mean(dim=0)  # global image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_global = F.normalize(image_features_global, dim=1)  
        image_features_local = image_features_local / image_features_local.norm(dim=-1, keepdim=True)
        
        prompts = self.prompt_learner()  # learnable prompts embedding
        n_cls = int(prompts.shape[0] / self.N)
        tokenized_prompts = self.tokenized_prompts  
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, n_cls, -1) 
        text_features_global = text_features.mean(dim=0)  # global text features

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_global = F.normalize(text_features_global, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous() 
        sim = sim.view(M, self.N, batch_size * n_cls)
        sim = sim.permute(2,0,1)  # shape: (batch_size*n_cls, M, N)
        
        sim_local = torch.einsum('mbtd,ncd->mnbtc', image_features_local, text_features).contiguous() 
        sim_local = sim_local.view(M, self.N, sim_local.shape[-2], batch_size * n_cls)
        sim_local = sim_local.permute(3, 0, 1, 2) 
        sim_local = sim_local.mean(dim=-1)  # shape: (batch_size*n_cls, M, N)

        scale_param = 0.125
        sim_op = torch.sum(sim * scale_param, dim=(1,2)) 
        sim_op_local = torch.sum(sim_local * scale_param, dim=(1,2))  

        sim_op = sim_op.contiguous().view(batch_size, n_cls)
        sim_op_local = sim_op_local.contiguous().view(batch_size, n_cls)
        
        logit_scale = self.logit_scale.exp()
        logit_scale_local = logit_scale * 1e-2
        logits = logit_scale * sim_op
        logits_local = logit_scale_local * sim_op_local
        
        logits += logits_local

        if label is not None:
            loss = F.cross_entropy(logits, label)
            return loss, logits, image_features_global, text_features_global
        else:
            return None, logits, image_features_global, text_features_global


    def forward(self, image, label=None):  

        loss, logits, _, _ = self.forward_image_text(image, label)

        if self.prompt_learner.training:
            if self.cfg.TRAINER.XCoOp.ENABLE:
                loss += self.cfg.TRAINER.XCoOp.XCoOp_LOSS_WEIGHT * self.forward_text_to_text_clinical()  # NOTE: clinical prompt-level
                loss += self.cfg.TRAINER.XCoOp.XCoOp_TOKEN_LOSS_WEIGHT * self.forward_token_to_token_clinical()  # NOTE: token-level loss

        if self.prompt_learner.training:
            return loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class XCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.XCoOp.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames 
        all_classnames = self.dm.dataset.all_class_names

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.XCoOp.PREC == "fp32" or cfg.TRAINER.XCoOp.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, all_classnames)

        print("Turning off not-required gradients")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):  
                param.requires_grad_(True)  # follow LASP

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        group1, group2 = [], []
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                group1.append(param)
            else:
                group2.append(param)

        param_groups = [{"params": group1, "lr": cfg.OPTIM.LR * 0.1},
                        {"params": group2},]
        self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.XCoOp.PREC == "amp" else None

        # wandb initialization
        if self.cfg.WANDB:
            run = wandb.init(
            project="XCoOp",
            config=self.cfg,
            name="Dataset_{}_Trainer_{}_Backbone_{}_epochs_{}_lr_{}".format(
                self.cfg.DATASET.NAME, self.cfg.TRAINER.NAME, self.cfg.MODEL.BACKBONE.NAME, self.cfg.OPTIM.MAX_EPOCH, self.cfg.OPTIM.LR)
            )

        # use to record training
        self.batch_losses = []

    def forward_backward(self, batch):
        
        image, label = self.parse_batch_train(batch)

        model = self.model  # CustomCLIP
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.XCoOp.PREC  # "amp"
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        self.batch_losses.append(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

            # wandb record the epoch loss
            epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
            if self.cfg.WANDB:
                wandb.log({"epoch_train_loss": epoch_loss})
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if isinstance(input, list):
            input = [inp.to(self.device, non_blocking=True) for inp in input]
        else:
            input = input.to(self.device, non_blocking=True)
        label = label.to(self.device)

        if self.cfg.DATALOADER.K_TRANSFORMS > 1:
            input = torch.cat(input)
            label = label.repeat(self.cfg.DATALOADER.K_TRANSFORMS)
        return input, label

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

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            ignore_list = ['token_prefix', 'token_suffix', 'token_prefix_all', 'token_suffix_all', 'class_text_features']
            ignore_list += [f'prompt_learner.{il}' for il in ignore_list]

            for k in ignore_list:
                state_dict.pop(k, None)

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            w_weights = None
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in self._models[name].state_dict():
                    if v.size() == self._models[name].state_dict()[k].size():
                        new_state_dict[k] = v
                    else:
                        print(k, v.shape, self._models[name].state_dict()[k].size())
            print(f'Num of preserved keys: {len(new_state_dict)}')
            print(f'Keys: {new_state_dict.keys()}')
            self._models[name].load_state_dict(new_state_dict, strict=False)
        return w_weights
    
    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)  # input.shape: (batch_size, 3, 224, 224)  label.shape: (batch_size,)
            output = self.model_inference(input)  # the logits of model, output.shape: (batch_size, num_classes)
            self.evaluator.process(output, label)  
        
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        # wandb record
        if self.cfg.WANDB:
            if split == "test":
                for k, v in results.items():
                    if k == "accuracy":
                        wandb.log({"test_acc": v})

        with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        return list(results.values())[0]

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            test_result = self.test(split="test")