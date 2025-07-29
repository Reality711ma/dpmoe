import torch.nn.functional as F
import torch
def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)

def class_loss(visual_features, class_prototypes, labels=None, t=0.07):
    # print(visual_features.shape, class_prototypes.shape, transpose(class_prototypes).shape)
    # logits = t.exp() * visual_features @ transpose(class_prototypes)
    logits = t.exp() * torch.einsum('bd, bcd->bc', visual_features, class_prototypes)
    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits


def domain_loss(visual_features, domain_text_features_all, labels=None, domains=None, t=0.07):
    batch_size = visual_features.size(0)
    num_domains = len(domain_text_features_all)
    # num_text_features = domain_text_features_all.size(1)

    # 初始化存储容器
    positive_samples = []
    negative_samples = []

    for i in range(batch_size):
        nega = []
        domain_idx = domains[i]

        positive_samples.append(domain_text_features_all[domain_idx])


    pos_sim = t.exp() * torch.einsum('bd, bcd->bc', visual_features, torch.stack(positive_samples))
    # neg_sim = torch.einsum('bd, bmcd->bmc', visual_features, torch.stack(negative_samples))


    return F.cross_entropy(pos_sim, labels)


