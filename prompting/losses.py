import torch.nn.functional as F

def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)

def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    logits = t.exp() * visual_features @ transpose(class_prototypes)
    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits

def contrastive_loss_token_level(soft_token_embeddings, hard_token_embeddings, labels=None, t=0.07):
    soft_token_embeddings = soft_token_embeddings.view(soft_token_embeddings.shape[0], soft_token_embeddings.shape[1], -1)
    hard_token_embeddings = hard_token_embeddings.view(hard_token_embeddings.shape[0], hard_token_embeddings.shape[1], -1)
    return contrastive_loss(soft_token_embeddings, hard_token_embeddings, labels=labels, t=t)

