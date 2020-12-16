# -------------------------------------------------------------------------------------
# A Bidirectional Focal Atention Network implementation based on
# https://arxiv.org/abs/1909.11416.
# "Focus Your Atention: A Bidirectional Focal Atention Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2019
# -------------------------------------------------------------------------------------
"""BFAN model"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from GCT import GCT
import MPNCOV
from MPNCOV import CovpoolLayer,SqrtmLayer,TriuvecLayer
from sklearn.decomposition import PCA
from GCN_lib.Rs_GCN import Rs_GCN

def l2normN(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError(
            "Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.GCT = GCT(36)
        self.init_weights()
        # self.ca = ChannelAttention(36)
        # self.fci = nn.Linear(666, 64)



    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
		# print('image')
		# print(features.size())
        features1= self.GCT(features)

        # print('image')  
        y = CovpoolLayer(features1)
        # print(y.size())                             #   128,36,36
        y = SqrtmLayer(y, 5)
        # print(y.size())                             #    128,36,36
        y = TriuvecLayer(y)                           #     128,666,1
        # print(y.size())      
        y = y.view(y.size(0), -1)  
        # # print(y.size())
        # y = self.fci(y)
        # # print(y.size())
		
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features, y

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :int(cap_emb.size(2)/2)] +cap_emb[:, :, int(cap_emb.size(2)/2):])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
		# print('text')
		# print(cap_emb.size())

        # print('text')  
        y = CovpoolLayer(cap_emb)
        # print(y.size())                             #   128,36,36
        y = SqrtmLayer(y, 5)
        # print(y.size())                             #    128,36,36
        y = TriuvecLayer(y)                           #     128,666,1
        # print(y.size())   
        channel =y.size(1)	
        # print(channel) 
        y = y.view(y.size(0), -1)  
        # print(y.size()) 


        # self.fct = nn.Linear(channel, 64).cuda()
        # y = self.fct(y)
		
        # print(y.size()) 

        return cap_emb, cap_len,y


def func_attention(query, context, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif opt.focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        raise ValueError("unknown focal attention type:", opt.focal_type)

    funcH=focal_equal(attn, batch_size, queryL, sourceL)+focal_prob(attn, batch_size, queryL, sourceL)

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt 
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size*queryL, sourceL, 1)
    xj = xj.view(batch_size*queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size*queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1-term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(cap_i_expand, images, opt)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(images, cap_i_expand, opt)
        i2t_sim = cosine_similarity(images, weiContext, dim=2)
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)

        # Overall similarity for image and text
        sim = t2i_sim + i2t_sim
        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        scores = xattn_score(im, s, s_l, self.opt)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

# class ContrastiveLoss2(nn.Module):
    # """
    # Compute contrastive loss
    # """

    # def __init__(self, margin=0, measure=False, max_violation=False):
        # super(ContrastiveLoss2, self).__init__()
        # self.margin = margin
        # if measure == 'order':
            # self.sim = order_sim
        # else:
            # self.sim = cosine_sim
        # # self.sim = cosine_sim
        # self.max_violation = max_violation

    # def forward(self, im, s):
        # # compute image-sentence score matrix
        # scores = self.sim(im, s)
        # diagonal = scores.diag().view(im.size(0), 1)
        # d1 = diagonal.expand_as(scores)
        # d2 = diagonal.t().expand_as(scores)

        # # compare every diagonal score to scores in its column
        # # caption retrieval
        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # # compare every diagonal score to scores in its row
        # # image retrieval
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = Variable(mask)
        # if torch.cuda.is_available():
            # I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # # keep the maximum violating negative for each query
        # if self.max_violation:
            # cost_s = cost_s.max(1)[0]
            # cost_im = cost_im.max(0)[0]

        # return cost_s.sum() + cost_im.sum()


# '''KL regularizer for softmax prob distribution'''
# class KL_loss_softmax(nn.Module):
    # """
    # Compute KL_divergence between all prediction score (already sum=1, omit softmax function)
    # """
    # def __init__(self):
        # super(KL_loss_softmax, self).__init__()

        # self.KL_loss = nn.KLDivLoss(reduce=False)

    # def forward(self, im, s):
        # img_prob = torch.log(im)
        # s_prob = s
        # KL_loss = self.KL_loss(img_prob, s_prob)
        # loss = KL_loss.sum()

        # return loss

# class ContrastiveLoss(nn.Module):
    # """
    # Compute contrastive loss
    # """
    # def __init__(self, opt, margin=0, max_violation=False):
        # super(ContrastiveLoss, self).__init__()
        # self.opt = opt
        # self.margin = margin
        # self.max_violation = max_violation
    # def forward(self, im, s, s_l):
        # # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
            # scores = 0.5*xattn_score_t2i(im, s, s_l, self.opt)+0.5*xattn_score_i2t(im, s, s_l, self.opt)
            # # scores = xattn_score_t2i(im, s, s_l, self.opt)
            # # scores = xattn_score_i2t(im, s, s_l, self.opt)
        # elif self.opt.cross_attn == 'i2t':
            # scores = xattn_score_i2t(im, s, s_l, self.opt)
        # else:
            # raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        # diagonal = scores.diag().view(im.size(0), 1)
        # d1 = diagonal.expand_as(scores)
        # d2 = diagonal.t().expand_as(scores)

        # # compare every diagonal score to scores in its column
        # # caption retrieval
        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # # compare every diagonal score to scores in its row
        # # image retrieval
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = Variable(mask)
        # if torch.cuda.is_available():
            # I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # # keep the maximum violating negative for each query
        # if self.max_violation:
            # cost_s = cost_s.max(1)[0]
            # cost_im = cost_im.max(0)[0]
        # return cost_s.sum() + cost_im.sum()


# '''Fusing instance-level feature and consensus-level feature'''
# class Multi_feature_fusing(nn.Module):
    # """
    # Emb the features from both modalities to the joint attribute label space.
    # """

    # def __init__(self, embed_dim, fuse_type='weight_sum'):
        # """
        # param image_dim: dim of visual feature
        # param embed_dim: dim of embedding space
        # """
        # super(Multi_feature_fusing, self).__init__()

        # self.fuse_type = fuse_type
        # self.embed_dim = embed_dim
        # if fuse_type == 'concat':
            # input_dim = int(2*embed_dim)
            # self.joint_emb_v = nn.Linear(input_dim, embed_dim)
            # self.joint_emb_t = nn.Linear(input_dim, embed_dim)
            # self.init_weights_concat()
        # if fuse_type == 'adap_sum':
            # self.joint_emb_v = nn.Linear(embed_dim, 1)
            # self.joint_emb_t = nn.Linear(embed_dim, 1)
            # self.init_weights_adap_sum()

    # def init_weights_concat(self):
        # """Xavier initialization"""
        # r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        # self.joint_emb_v.weight.data.uniform_(-r, r)
        # self.joint_emb_v.bias.data.fill_(0)
        # self.joint_emb_t.weight.data.uniform_(-r, r)
        # self.joint_emb_t.bias.data.fill_(0)

    # def init_weights_adap_sum(self):
        # """Xavier initialization"""
        # r = np.sqrt(6.) / np.sqrt(self.embed_dim + 1)
        # self.joint_emb_v.weight.data.uniform_(-r, r)
        # self.joint_emb_v.bias.data.fill_(0)
        # self.joint_emb_t.weight.data.uniform_(-r, r)
        # self.joint_emb_t.bias.data.fill_(0)

    # def forward(self, v_emb_instance, t_emb_instance, v_emb_concept, t_emb_concept):
        # """
        # Forward propagation.
        # :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        # :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        # :return: joint embbeding features for both modalities
        # """
        # if self.fuse_type == 'multiple':
            # v_fused_emb = v_emb_instance.mul(v_emb_concept);
            # v_fused_emb = l2normX(v_fused_emb)
            # t_fused_emb = t_emb_instance.mul(t_emb_concept);
            # t_fused_emb = l2normX(t_fused_emb)

        # elif self.fuse_type == 'concat':
            # v_fused_emb = torch.cat([v_emb_instance, v_emb_concept], dim=1)
            # v_fused_emb = self.joint_emb_instance_v(v_fused_emb)
            # v_fused_emb = l2normX(v_fused_emb)

            # t_fused_emb = torch.cat([t_emb_instance, t_emb_concept], dim=1)
            # t_fused_emb = self.joint_emb_instance_v(t_fused_emb)
            # t_fused_emb = l2normX(t_fused_emb)

        # elif self.fuse_type == 'adap_sum':
            # v_mean = (v_emb_instance + v_emb_concept) / 2
            # v_emb_instance_mat = self.joint_emb_instance_v(v_mean)
            # alpha_v = F.sigmoid(v_emb_instance_mat)
            # v_fused_emb = alpha_v * v_emb_instance + (1 - alpha_v) * v_emb_concept
            # v_fused_emb = l2normX(v_fused_emb)

            # t_mean = (t_emb_instance + t_emb_concept) / 2
            # t_emb_instance_mat = self.joint_emb_instance_t(t_mean)
            # alpha_t = F.sigmoid(t_emb_instance_mat)
            # t_fused_emb = alpha_t * t_emb_instance + (1 - alpha_t) * t_emb_concept
            # t_fused_emb = l2normX(t_fused_emb)

        # elif self.fuse_type == 'weight_sum':
            # alpha = 0.75
            # v_fused_emb = alpha * v_emb_instance + (1 - alpha) * v_emb_concept
            # v_fused_emb = l2normX(v_fused_emb)
            # t_fused_emb = alpha * t_emb_instance + (1 - alpha) * t_emb_concept
            # t_fused_emb = l2normX(t_fused_emb)

        # return v_fused_emb, t_fused_emb

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class SPConv_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
        super(SPConv_3x3, self).__init__()
        self.inplanes_3x3 = int(inplanes*ratio)
        self.inplanes_1x1 = inplanes - self.inplanes_3x3
        self.outplanes_3x3 = int(outplanes*ratio)
        self.outplanes_1x1 = outplanes - self.outplanes_3x3
        self.outplanes = outplanes
        self.stride = stride

        self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
                             padding=1, groups=2, bias=False)
        self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)

        self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes,kernel_size=1)
        self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.ratio = ratio
        self.groups = int(1/self.ratio)
    def forward(self, x):
        b, c, _, _ = x.size()


        x_3x3 = x[:,:int(c*self.ratio),:,:]
        x_1x1 = x[:,int(c*self.ratio):,:,:]
        out_3x3_gwc = self.gwc(x_3x3)
        if self.stride ==2:
            x_3x3 = self.avgpool_s2_3(x_3x3)
        out_3x3_pwc = self.pwc(x_3x3)
        out_3x3 = out_3x3_gwc + out_3x3_pwc
        out_3x3 = self.bn1(out_3x3)
        out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # use avgpool first to reduce information lost
        if self.stride == 2:
            x_1x1 = self.avgpool_s2_1(x_1x1)

        out_1x1 = self.conv1x1(x_1x1)
        out_1x1 = self.bn2(out_1x1)
        out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        out = out_1x1 * (out_31_ratio[:,:,1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
              + out_3x3 * (out_31_ratio[:,:,0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

        return out


class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x

# class GCT(nn.Module):

    # def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        # super(GCT, self).__init__()

        # self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        # self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        # self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        # self.epsilon = epsilon
        # self.mode = mode
        # self.after_relu = after_relu

    # def forward(self, x):

        # if self.mode == 'l2':
            # embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            # norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        # elif self.mode == 'l1':
            # if not self.after_relu:
                # _x = torch.abs(x)
            # else:
                # _x = x
            # embedding = _x.sum((2,3), keepdim=True) * self.alpha
            # norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        # else:
            # print('Unknown mode!')
            # sys.exit()

        # gate = 1. + torch.tanh(embedding * norm + self.beta)
        # out = x * gate
        # return out


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def l2normX(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


''' Visual self-attention module '''
class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, use_bn, activation_type, dropout_rate, img_region_num):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(embed_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2normX((v_att))

        return v_att, weights


''' Textual self-attention module '''
class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, use_bn, activation_type, dropout_rate):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """

        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2normX(u_att)

        return u_att, weights


'''Fusing instance-level feature and consensus-level feature'''
class Multi_feature_fusing(nn.Module):
    """
    Emb the features from both modalities to the joint attribute label space.
    """

    def __init__(self, embed_dim, fuse_type='weight_sum'):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Multi_feature_fusing, self).__init__()

        self.fuse_type = fuse_type
        self.embed_dim = embed_dim
        if fuse_type == 'concat':
            input_dim = int(2*embed_dim)
            self.joint_emb_v = nn.Linear(input_dim, embed_dim)
            self.joint_emb_t = nn.Linear(input_dim, embed_dim)
            self.init_weights_concat()
        if fuse_type == 'adap_sum':
            self.joint_emb_v = nn.Linear(embed_dim, 1)
            self.joint_emb_t = nn.Linear(embed_dim, 1)
            self.init_weights_adap_sum()

    def init_weights_concat(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def init_weights_adap_sum(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 1)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def forward(self, v_emb_instance, t_emb_instance, v_emb_concept, t_emb_concept):
        """
        Forward propagation.
        :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        :return: joint embbeding features for both modalities
        """
        if self.fuse_type == 'multiple':
            v_fused_emb = v_emb_instance.mul(v_emb_concept);
            v_fused_emb = l2normX(v_fused_emb)
            t_fused_emb = t_emb_instance.mul(t_emb_concept);
            t_fused_emb = l2normX(t_fused_emb)

        elif self.fuse_type == 'concat':
            v_fused_emb = torch.cat([v_emb_instance, v_emb_concept], dim=1)
            v_fused_emb = self.joint_emb_instance_v(v_fused_emb)
            v_fused_emb = l2normX(v_fused_emb)

            t_fused_emb = torch.cat([t_emb_instance, t_emb_concept], dim=1)
            t_fused_emb = self.joint_emb_instance_v(t_fused_emb)
            t_fused_emb = l2normX(t_fused_emb)

        elif self.fuse_type == 'adap_sum':
            v_mean = (v_emb_instance + v_emb_concept) / 2
            v_emb_instance_mat = self.joint_emb_instance_v(v_mean)
            alpha_v = F.sigmoid(v_emb_instance_mat)
            v_fused_emb = alpha_v * v_emb_instance + (1 - alpha_v) * v_emb_concept
            v_fused_emb = l2normX(v_fused_emb)

            t_mean = (t_emb_instance + t_emb_concept) / 2
            t_emb_instance_mat = self.joint_emb_instance_t(t_mean)
            alpha_t = F.sigmoid(t_emb_instance_mat)
            t_fused_emb = alpha_t * t_emb_instance + (1 - alpha_t) * t_emb_concept
            t_fused_emb = l2normX(t_fused_emb)

        elif self.fuse_type == 'weight_sum':
            alpha = 0.75
            v_fused_emb = alpha * v_emb_instance + (1 - alpha) * v_emb_concept
            v_fused_emb = l2normX(v_fused_emb)
            t_fused_emb = alpha * t_emb_instance + (1 - alpha) * t_emb_concept
            t_fused_emb = l2normX(t_fused_emb)

        return v_fused_emb, t_fused_emb


class Multi_fusing(nn.Module):
    """
    Emb the features from both modalities to the joint attribute label space.
    """

    def __init__(self, embed_dim):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Multi_fusing, self).__init__()

        # self.image_dim = image_dim
        # self.text_dim = text_dim
        self.embed_dim = embed_dim

        self.img_rnn = nn.GRU(embed_dim, embed_dim, 1, batch_first=True, bidirectional=True)

        # GCN reasoning 
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_dim, inter_channels=embed_dim)
		


    def init_weights_concat(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        self.joint_emb.weight.data.uniform_(-r, r)
        self.joint_emb.bias.data.fill_(0)

    def var_PCA(self,var,t_size):
        pca = PCA(t_size)
        #print(img_emb1.shape)
        temp_tensor=torch.Tensor(var.shape[0],t_size,var.shape[2]).zero_()
        for i in range(var.shape[0]):
            temp=var[i].reshape(var[i].shape[1],var[i].shape[0])
            l=temp.cpu().detach().numpy()
            l[np.isnan(l)]=0
            l=pca.fit_transform(l)
            temp = torch.from_numpy(l)
            temp_tensor[i]=temp.reshape(t_size,var[i].shape[1])
        var=temp_tensor.cuda()
        return var


    def forward(self, fi, ft):
        """
        Forward propagation.
        :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        :return: joint embbeding features for both modalities
        """

        # fi = self.fci(fi)
        # ft = self.fct(ft)
        # print(fi.size())
        # print(ft.size())
        fiold=fi
        ftold=ft
        fio=fi
        fto=ft
        # # fused_emb = torch.cat([fi, ft], dim=1)
        # # print(fused_emb.size())
        # # fused_emb = self.joint_emb(fused_emb)
        # # # fused_emb = l2normX(fused_emb)
        # # print(fused_emb.size())
        # CUDA=torch.device('cuda')
        # bn = nn.InstanceNorm1d(num_features=3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # ln = nn.LayerNorm(normalized_shape=ft.size()[1:], eps=0, elementwise_affine=False)
        # Fusion = torch.cat((fi, ft),dim=1)
        # dn=torch.nn.Linear(1024, 1024)
        # dn = dn.cuda(CUDA)
        # Fusion=Variable(Fusion) 

        # Fusion_to_text=Fusion.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # score=cosine_similarity(Fusion_to_text,ft,dim=1)
        # score=Variable(score)
        # score=score.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # Fusion_to_text=Fusion_to_text*score
        # Fusion_to_text=bn(Fusion_to_text)
        # Fusion_to_text=dn(Fusion_to_text)
        # ft = torch.cat((ft, Fusion_to_text))
        # ft=dn(ft)
        # ft=Variable(ft)
        # ft=ft.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # ft=Variable(ft,requires_grad=True)

        # Fusion_to_image=Fusion.resize_(fi.shape[0], fi.shape[1], fi.shape[2])
        # Fusion=Variable(Fusion,requires_grad=True) 
        # #print(Fusion.shape,cap_emb.shape)
        # score=cosine_similarity(fi,Fusion_to_image,dim=1)
        # score=Variable(score)
        # score=score.resize_(fio.shape[0], fio.shape[1], fio.shape[2])
        # Fusion_to_image=Fusion_to_image*score
        # Fusion_to_image=bn(Fusion_to_image)
        # Fusion_to_image=dn(Fusion_to_image)
        # fi = torch.cat((fi, Fusion_to_image),dim=1)
        # fi=dn(fi)
        # fi=Variable(fi)
        # fi=fi.resize_(fi.shape[0], fi.shape[1], fi.shape[2])
        # fi=Variable(fi,requires_grad=True)

        # fi=bn(fi)
        # ft=ln(ft)

        # fi=self.var_PCA(fi,fio.shape[1])

        GCN_img_emd = fiold.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2normN(GCN_img_emd)


        GCN_txt_emd = ftold.permute(0, 2, 1)
        GCN_txt_emd = self.Rs_GCN_1(GCN_txt_emd)
        GCN_txt_emd = self.Rs_GCN_2(GCN_txt_emd)
        GCN_txt_emd = self.Rs_GCN_3(GCN_txt_emd)
        GCN_txt_emd = self.Rs_GCN_4(GCN_txt_emd)
        # -> B,N,D
        GCN_txt_emd = GCN_txt_emd.permute(0, 2, 1)

        GCN_txt_emd = l2normN(GCN_txt_emd)

        CUDA=torch.device('cuda')
        bn = nn.InstanceNorm1d(num_features=3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        ln = nn.LayerNorm(normalized_shape=ft.size()[1:], eps=0, elementwise_affine=False)
        Fusion = torch.cat((fi, ft),dim=1)
        dn=torch.nn.Linear(1024, 1024)
        dn = dn.cuda(CUDA)
        Fusion=Variable(Fusion) 
#################################################################################################
        # # Fusion_to_text=Fusion.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # # score=cosine_similarity(Fusion_to_text,ft,dim=1)
        # # score=Variable(score)
        # # score=score.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # # Fusion_to_text=Fusion_to_text*score

        # # Fusion_to_text=bn(Fusion_to_text)
        # Fusion_to_text=dn(Fusion_to_text)
        # ft = torch.cat((ft, Fusion_to_text))
        # ft=dn(ft)
        # ft=Variable(ft)
        # ft=ft.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        # ft=Variable(ft,requires_grad=True)

        # # Fusion_to_image=Fusion.resize_(fi.shape[0], fi.shape[1], fi.shape[2])
        # # Fusion=Variable(Fusion,requires_grad=True) 
        # # #print(Fusion.shape,cap_emb.shape)
        # # score=cosine_similarity(fi,Fusion_to_image,dim=1)
        # # score=Variable(score)
        # # score=score.resize_(fio.shape[0], fio.shape[1], fio.shape[2])
        # # Fusion_to_image=Fusion_to_image*score

        # # Fusion_to_image=bn(Fusion_to_image)
        # Fusion_to_image=dn(Fusion_to_image)
        # fi = torch.cat((fi, Fusion_to_image),dim=1)
        # fi=dn(fi)
        # fi=Variable(fi)
        # fi=fi.resize_(fi.shape[0], fi.shape[1], fi.shape[2])
        # fi=Variable(fi,requires_grad=True)

        # # fi=bn(fi)
        # # ft=ln(ft)

##########################################################################################################################33

        fio = fio.view(fio.size(0), fio.size(1), fio.size(2), 1).contiguous()  # Nx64x64x1
        fto = fto.view(fto.size(0), fto.size(1), fto.size(2), 1).contiguous()  # Nx64x64x1
        # CA= ChannelAttention(fio.size(1))
        spmi = SPBlock(inplanes=fio.size(1), outplanes=fio.size(1)).cuda()
        spmt = SPBlock(inplanes=fto.size(1), outplanes=fto.size(1)).cuda()
        fi = fio * spmi(fio)+fio
        ft = fto * spmt(fto)+fto
		
        # fi = fi.view(fi.size(0), -1)  
        # ft = ft.view(ft.size(0), -1) 
		
		
        fio = fio.view(fio.size(0), fio.size(1), fio.size(2), 1).contiguous()  # Nx64x64x1
        fto = fto.view(fto.size(0), fto.size(1), fto.size(2), 1).contiguous()  # Nx64x64x1
        # CA= ChannelAttention(fio.size(1))
        spmi = SPBlock(inplanes=fio.size(1), outplanes=fio.size(1)).cuda()
        spmt = SPBlock(inplanes=fto.size(1), outplanes=fto.size(1)).cuda()
        fi = fio * spmi(fio)+fio
        ft = fto * spmt(fto)+fto
		
        # fi = fi.view(fi.size(0), -1)  
        # ft = ft.view(ft.size(0), -1) 
		
		
        fi1 =fi.squeeze()
        ft1 =ft.squeeze()
		
        fi2 = torch.cat((fi1, fiold),dim=1)
        ft2 = torch.cat((ft1, ftold),dim=1)
		


        fi21 =fi2
        ft21 =ft2
        # print(fi21.size())
        # print(ft21.size())

        fi21 = fi21.view(fi21.size(0), fi21.size(1), fi21.size(2), 1).contiguous()  # Nx64x64x1
        ft21 = ft21.view(ft21.size(0), ft21.size(1), ft21.size(2), 1).contiguous()  # Nx64x64x1
        SPBi = SPBlock(inplanes=fi2.size(1), outplanes=fiold.size(1)).cuda()
        SPBt = SPBlock(inplanes=ft2.size(1), outplanes=ftold.size(1)).cuda()
        # print(SPBi(fi21).size())
        # print(SPBt(ft21).size())

        fiold1 = fiold.view(fiold.size(0), fiold.size(1), fiold.size(2), 1).contiguous()  # Nx64x64x1
        ftold1 = ftold.view(ftold.size(0), ftold.size(1), ftold.size(2), 1).contiguous()  # Nx64x64x1

        ffi = fiold1 * SPBi(fi21)+fiold1
        fft = ftold1 * SPBt(ft21)+ftold1



        ffi =ffi.squeeze()
        fft =fft.squeeze()
		

        yi = CovpoolLayer(fiold)
        # print(y.size())                             #   128,36,36
        yi = SqrtmLayer(yi, 5)
        # print(y.size())                             #    128,36,36
        yi = TriuvecLayer(yi)                           #     128,666,1
        # print(y.size())   
        channel =yi.size(1)	
        # print(channel) 
        yi = yi.view(yi.size(0), -1)  
        # print(y.size()) 
        fci = nn.Linear(channel, 256).cuda()
        yi = fci(yi)


        yt = CovpoolLayer(ftold)
        # print(y.size())                             #   128,36,36
        yt = SqrtmLayer(yt, 5)
        # print(y.size())                             #    128,36,36
        yt = TriuvecLayer(yt)                           #     128,666,1
        # print(y.size())   
        channel =yt.size(1)	
        # print(channel) 
        yt = yt.view(yt.size(0), -1)  
        # print(y.size()) 
        fct = nn.Linear(channel, 256).cuda()
        yt = fct(yt)


        F_I = F.normalize(yi)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1
        F_T = F.normalize(yt)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1
		
        S_tilde = 0.9 * S_I + (1 - 0.9) * S_T
        S = 1.5 * S_tilde

        Fusion_to_image=ffi
        Fusion_to_image=Variable(Fusion_to_image,requires_grad=True)
        Fusion_to_image=dn(Fusion_to_image)
        fi00 = torch.cat((fiold, Fusion_to_image),dim=1)
        fi00=dn(fi00)
        fi00=Variable(fi00)
        fi00=fi00.resize_(fi00.shape[0], fi00.shape[1], fi00.shape[2])
        fi00=Variable(fi00,requires_grad=True)

        Fusion_to_text=fft
        Fusion_to_text=Variable(Fusion_to_text,requires_grad=True)
        Fusion_to_text=dn(Fusion_to_text)
        ft00 = torch.cat((ftold, Fusion_to_text))
        ft00=dn(ft00)
        ft00=Variable(ft00)
        ft00=ft00.resize_(fto.shape[0], fto.shape[1], fto.shape[2])
        ft00=Variable(ft00,requires_grad=True)

        return fi00, ft00,yi, yt,S
		
		
class BFAN(object):
    """
    Bidirectional Focal Attention Network (BFAN) model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   no_txtnorm=opt.no_txtnorm)
        self.Multi_fusing = Multi_fusing(embed_dim=opt.embed_size)
        img_region_num = 36
        # visual self-attention
        self.V_self_atten_enhance = V_single_modal_atten(opt.embed_size, opt.embed_size, opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate, img_region_num)
        # textual self-attention
        self.T_self_atten_enhance = T_single_modal_atten(opt.embed_size, opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate)

        self.Multi_feature_fusing = Multi_feature_fusing(embed_dim=opt.embed_size, fuse_type=opt.feature_fuse_type)
        self.losstemp=True

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.Multi_fusing.cuda()
            self.V_self_atten_enhance.cuda()
            self.T_self_atten_enhance.cuda()
            self.Multi_feature_fusing.cuda()
            cudnn.benchmark = True

        # # Loss and Optimizer
        # self.criterion = ContrastiveLoss(opt=opt,
                                         # margin=opt.margin,
                                         # max_violation=opt.max_violation)

        # self.criterion_rank = ContrastiveLoss2(margin=opt.margin,
                                              # measure=opt.measure,
                                              # max_violation=opt.max_violation)
        # self.criterion_KL_softmax = KL_loss_softmax()


        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        # # Loss and Optimizer
        # self.criterion2 = ContrastiveLoss2(opt=opt,
                                         # margin=opt.margin,
                                         # max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.Multi_fusing.parameters())
        params += list(self.V_self_atten_enhance.parameters())
        params += list(self.T_self_atten_enhance.parameters())
        params += list(self.Multi_feature_fusing.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),self.Multi_fusing.state_dict(),
                      self.V_self_atten_enhance.state_dict(),
                      self.T_self_atten_enhance.state_dict(),
                      self.Multi_feature_fusing.state_dict()]

        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.Multi_fusing.load_state_dict(state_dict[2])
        self.V_self_atten_enhance.load_state_dict(state_dict[3])
        self.T_self_atten_enhance.load_state_dict(state_dict[4])
        self.Multi_feature_fusing.load_state_dict(state_dict[5])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.Multi_fusing.train()
        self.V_self_atten_enhance.train()
        self.T_self_atten_enhance.train()
        self.Multi_feature_fusing.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.Multi_fusing.eval()
        self.V_self_atten_enhance.eval()
        self.T_self_atten_enhance.eval()
        self.Multi_feature_fusing.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb,fi = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens,ft = self.txt_enc(captions, lengths)


        orimg_emb=img_emb
        orcap_emb=cap_emb

        img_emb, cap_emb,yi, yt,S= self.Multi_fusing(img_emb, cap_emb)
		
        # img_emb_mean = torch.mean(img_emb, 1)
        # cap_emb_mean = torch.mean(cap_emb, 1)

        # instance_emb_v0, visual_weights0 = self.V_self_atten_enhance(img_emb, img_emb_mean)
        # instance_emb_t0, textual_weights0 = self.T_self_atten_enhance(cap_emb, cap_emb_mean)

        return img_emb, cap_emb, cap_lens,orimg_emb,orcap_emb,yi, yt,S

    # def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        # """Compute the loss given pairs of image and caption embeddings
        # """
        # self.optimizer.zero_grad()
        # loss = self.criterion(img_emb, cap_emb, cap_len)
        # self.logger.update('Le', loss.item())
        # return loss

    # def forward_loss2(self,fi, ft, **kwargs):#, bboxes, depends
        # self.optimizer.zero_grad()
        # loss =self.criterion_rank(fi, ft)
        # return loss

    def forward_loss(self, img_emb, cap_emb, cap_lens,orimg_emb,orcap_emb,yi, yt,S, **kwargs):#, bboxes, depends
        """Compute the loss given pairs of image and caption embeddings
        """
        # if self.losstemp==False:
            # loss = self.criterion(orimg_emb, orcap_emb, cap_lens)
			# #self.criterion2(orimg_emb, orcap_emb, cap_len)+
            # # loss = self.criterion(orimg_emb, orcap_emb, cap_len, ids)
            # # loss2 = self.criterion2(img_emb, cap_emb, cap_len, ids)			
            # #print("F")
            # self.logger.update('LeOld', loss.item(), img_emb.size(0))
            # self.losstemp=True
        # else:
            # loss = self.criterion(img_emb, cap_emb, cap_lens)+self.criterion(orimg_emb, orcap_emb, cap_lens)#+self.criterion(GCN_img_emd, orcap_emb, cap_len)#+self.forward_lossG(self.forward_sim(img_emb, cap_emb, bboxes, depends, cap_len))#
            # #print("T")
            # self.logger.update('LeOldNew', loss.item(), img_emb.size(0))
            # self.losstemp=False

        # # try:
            # # self.logger.update('Le', loss.data[0], img_emb.size(0))
        # # except:
            # # self.logger.update('Le', loss.data, img_emb.size(0))
        loss = self.criterion(orimg_emb, orcap_emb, cap_lens)
        self.logger.update('LeOld', loss.item(), img_emb.size(0))
        return loss


    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens,orimg_emb,orcap_emb,yi, yt,S = self.forward_emb(images, captions, lengths)

        all_1 = torch.rand((orimg_emb.size(0))).fill_(1).cuda()
        diagonal = S.diagonal()
        loss4 = F.mse_loss(diagonal, 1.5 * all_1)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens,orimg_emb,orcap_emb,yi, yt,S)+loss4
        self.logger.update('LeALL', loss.item(), img_emb.size(0))
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
