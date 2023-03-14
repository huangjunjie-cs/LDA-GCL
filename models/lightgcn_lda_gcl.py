#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# @Time   : 2021/09/01
# @Author : Junjie Huang
# @Email  : junjiehuang.cs@gmail.com
import time
import torch
import torch_sparse

import numpy as np
import scipy.sparse as sp

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage



from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F



class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm).pow(2)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


# https://github.com/wujcan/SGL/blob/main/model/general_recommender/SGL.py

def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample




class LDAGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, model0=None, vals=None, inds=None):
        super().__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.config=config
        self.dataset=dataset
        self.tensorboard = get_tensorboard(self.logger)

        self.iter_step = 0
        self.iter_step1 = 0
       
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers   = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

         # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss  = BPRLoss()
        self.reg_loss = EmbLoss()

        self.ssl_rm_ratio  = config['ssl_rm_ratio'] if 'ssl_rm_ratio' in config else 0.1 # get if from config TODO
        self.ssl_add_ratio = config['ssl_add_ratio'] if 'ssl_add_ratio' in config else 0.0 # get if from config TODO

        self.learn_flag = config['learn_flag'] if 'learn_flag' in config else True

        self.ssl_reg   = config['ssl_reg'] if 'ssl_reg' in config else 0.1
        self.ssl_temp  = config['ssl_temp'] if 'ssl_temp' in config else 0.2
        self.sub_mat   = defaultdict(list)

        self.adj = self.get_origin_adj()
        # self.ui_score = self.get_condidate_from_UUCF()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.lightgcn = model0
        self.vals = vals
        self.inds = inds

        self.init_neg_list()

        self.adv_reg  = config['adv_reg'] if 'adv_reg' in config else 0.2
        self.lightgcn = None

    def init_neg_list(self):
        training_user, training_item = self.get_user_item_list()
        user_np = np.array(training_user)
        item_np = np.array(training_item)
        pos_list = np.concatenate([[user_np, item_np]], axis=1)
        self.pos_list = pos_list.transpose()
        sumA = self.adj.sum(axis=1)
        topk_arr = sumA
        neg_list = []
        for index, v in enumerate(topk_arr[:self.n_users]):
            topk = int(v[0])
            if topk > 0:
                ind2 = self.inds[index, :topk]
                ind1 = torch.ones_like(ind2).long() * index
                ind1 = ind1.unsqueeze(1)
                ind2 = ind2.unsqueeze(1)
                ind3 = torch.cat([ind1, ind2], dim=1)
                neg_list.append(ind3)
        self.neg_list = torch.cat(neg_list, dim=0).cpu().numpy()
        self.edge_list = np.concatenate([self.pos_list, self.neg_list], axis=0)


    def init_lightgcn(self):
        if True:
        # if not self.lightgcn is None:
            params = {i:j for i,j in self.lightgcn.named_parameters()}
            self.user_embedding.weight.data.copy_(params['user_embedding.weight'])
            self.item_embedding.weight.data.copy_(params['item_embedding.weight'])

            self.lightgcn = None


    def sparse_mm(self, g, emb):
        N = self.n_users + self.n_items
        ind, weight = g
        return torch_sparse.spmm(ind.t(), weight, N, N, emb)


    def get_user_item_list(self):
        dok_matrix = self.interaction_matrix.todok()
        users_list, items_list = [], []
        for (user, item), value in dok_matrix.items():
            users_list.append(user)
            items_list.append(item)
        # (users_np, items_np) = self.train_matrix.nonzero()
        return users_list, items_list

    def get_origin_adj(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        return A
    

    def get_condidate_from_UUCF(self):
        # self.adj
        A_2 = self.adj
        A_2 = A_2.dot(self.adj) # u-u
        A_2.setdiag(0)
        A_3 = A_2.dot(self.adj) # u-i
        return A_3
    

    def create_adj_mat(self, edge_policy):
        training_user, training_item = self.get_user_item_list()

        if not self.learn_flag:
            drop_ratio = self.ssl_rm_ratio
            keep_idx = randint_choice(len(training_user), size=int(len(training_user) * (1 - drop_ratio)), replace=False)
            user_np = np.array(training_user)[keep_idx]
            item_np = np.array(training_item)[keep_idx]

            edge_list = np.concatenate([[user_np], [item_np]]).transpose()
            assert edge_list.shape[1] == 2
            ind = torch.from_numpy(edge_list).long().to(self.device) 
            weight = torch.ones(ind.shape[0]).to(self.device)
            
            if self.ssl_add_ratio > 0:
                add_ratio = self.ssl_add_ratio
                sumA = self.adj.sum(axis=1)
                topk_arr = sumA * add_ratio # 
                neg_list = self.neg_list
                keep_idx = randint_choice(len(neg_list), size=int(len(neg_list) *self.ssl_add_ratio), replace=False)
                user_np = neg_list[keep_idx][:, 0]
                item_np = neg_list[keep_idx][:, 1]
                edge_list_2 = np.concatenate([[user_np], [item_np]]).transpose()
                ind2 = torch.from_numpy(edge_list_2).long().to(self.device) 
                weight2 = torch.ones(ind2.shape[0]).to(self.device)
                ind = torch.cat([ind, ind2], dim=0)
                weight = torch.cat([weight, weight2], dim=0)

        else:
            edge_list = self.edge_list 
            pos_list = self.pos_list
            neg_list = self.neg_list

            user_all_embeddings, item_all_embeddings = self.forward()
            flags_1 = torch.ones(pos_list.shape[0], 1).to(self.device)
            flags_0 = torch.zeros(neg_list.shape[0], 1).to(self.device)
            flags = torch.cat([flags_1, flags_0], dim=0)
            u_emb = user_all_embeddings[edge_list[:, 0]]
            i_emb = item_all_embeddings[edge_list[:, 1]]
            inputs = torch.cat([u_emb, i_emb, flags], dim=1)
            
            weight = edge_policy(inputs).squeeze()
            ind = torch.from_numpy(edge_list).long().to(self.device) 
        assert ind.shape[1] == 2, f'ind shape {ind.shape}'
        assert ind[:, 1].max() < (self.n_users + self.n_items), f'c max:{ind[:,1].max()}'

        r = ind[:, 0] 
        c = ind[:, 1] + self.n_users
        N = self.n_users + self.n_items
        row = torch.cat([r, c], dim=0).unsqueeze(-1)
        col = torch.cat([c, r], dim=0).unsqueeze(-1)
        ind = torch.cat([row, col], dim=1)
        weight = torch.cat([weight, weight], dim=0)

        if not edge_policy.training:
           weight = weight.detach() 
        
        sumArr = torch_sparse.spmm(ind.t(), weight, N, N, torch.ones(N, 1).to(self.device))
        diag = sumArr + 1e-7
        diag = diag.pow(-0.5)
        w1 = diag[ind[:, 0]].squeeze()
        w2 = diag[ind[:, 1]].squeeze()
        weight = weight * w1 * w2
        return ind, weight


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def create_lightgcn_ssl_embed(self):
        for k in range(1, self.n_layers+1):
            self.sub_mat['sub_mat_1%d' % k] = self.sub_mat['sub_mat1']
            self.sub_mat['sub_mat_2%d' % k] = self.sub_mat['sub_mat2']

        ego_embeddings = self.get_ego_embeddings()

        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        for k in range(1, self.n_layers + 1):
            ego_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings)
            all_embeddings += [ego_embeddings]

            tmp_a = self.sub_mat['sub_mat_1%d' % k]
            ego_embeddings_sub1 = torch.sparse.mm(tmp_a, ego_embeddings_sub1)

            all_embeddings_sub1 += [ego_embeddings_sub1]

            g = self.sub_mat['sub_mat_2%d' % k]
            ego_embeddings_sub2 = self.sparse_mm(g, ego_embeddings_sub2)

            all_embeddings_sub2 += [ego_embeddings_sub2]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, axis=1)
        assert all_embeddings.size() == (self.n_users+self.n_items, self.latent_dim)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        all_embeddings_sub1 = torch.stack(all_embeddings_sub1, 1)
        all_embeddings_sub1 = torch.mean(all_embeddings_sub1, dim=1)
        u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(all_embeddings_sub1, [self.n_users, self.n_items])
        
        all_embeddings_sub2 = torch.stack(all_embeddings_sub2, 1)
        all_embeddings_sub2 = torch.mean(all_embeddings_sub2, dim=1)
        u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(all_embeddings_sub2, [self.n_users, self.n_items])

        return u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2

    def calc_ssl_loss_v0(self, interaction, u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2):
        '''
        Calculating SSL loss
        '''

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

        user_emb1 = u_g_embeddings_sub1[user]
        user_emb2 = u_g_embeddings_sub2[user]

        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(u_g_embeddings_sub2, dim=1)

        pos_score_user = torch.einsum("ij, ij->i", [normalize_user_emb1, normalize_user_emb2])
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)

        ttl_score_user = torch.einsum("ij, kj->ik", [normalize_user_emb1, normalize_all_user_emb2])
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), axis=1)
        
        ssl_loss_user = -torch.mean(torch.log(pos_score_user / ttl_score_user))


        item_emb1 = i_g_embeddings_sub1[pos_item]
        item_emb2 = i_g_embeddings_sub2[pos_item]

        normalize_item_emb1 = F.normalize(item_emb1, dim=1)
        normalize_item_emb2 = F.normalize(item_emb2, dim=1)
        normalize_all_item_emb2 = F.normalize(i_g_embeddings_sub2, dim=1)

        pos_score_item = torch.einsum("ij, ij->i", [normalize_item_emb1, normalize_item_emb2])
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)

        ttl_score_item = torch.einsum("ij, kj->ik", [normalize_item_emb1, normalize_all_item_emb2])
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), axis=1)

        ssl_loss_item = -torch.mean(torch.log(pos_score_item / ttl_score_item))

        ssl_loss = ssl_loss_user + ssl_loss_item
        return ssl_loss

    def calculate_ssl_loss(self, interaction, edge_policy, epoch_idx=0):
        if epoch_idx == 0:
            self.sub_mat['sub_mat1'] = self.norm_adj_matrix
            self.sub_mat['sub_mat2'] = self.create_adj_mat(edge_policy)

        t1 = time.time()
        # calculate SSL Loss
        u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2 = self.create_lightgcn_ssl_embed()
        ssl_loss = self.calc_ssl_loss_v0(interaction, u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2)
        l1 = -1 * ssl_loss * self.ssl_reg

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        u_embeddings   = u_g_embeddings_sub2[user]
        pos_embeddings = i_g_embeddings_sub2[pos_item]
        neg_embeddings = i_g_embeddings_sub2[neg_item]
    
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        if epoch_idx == 0:
            ind, weight = self.sub_mat['sub_mat2']
            mat = self.sub_mat['sub_mat1']
            sum_v = mat.coalesce().values().sum()
            self.iter_step += 1
            self.tensorboard.add_scalar('ssl mf loss', mf_loss.item(), self.iter_step)
            self.tensorboard.add_scalar('ssl loss', l1.item(), self.iter_step)
            self.tensorboard.add_scalar('sub mat1 sum', sum_v.item(), self.iter_step)
            self.tensorboard.add_scalar('sub mat2 sum', weight.sum().item(), self.iter_step)

        return mf_loss + l1 * self.adv_reg
        


    def calculate_loss(self, interaction, edge_policy, epoch_idx=0):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        if epoch_idx == 0:
            self.sub_mat['sub_mat1'] = self.norm_adj_matrix
            # self.sub_mat['sub_mat2'] = self.norm_adj_matrix
            self.sub_mat['sub_mat2'] = self.create_adj_mat(edge_policy=edge_policy)

        # calculate SSL Loss
        u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2 = self.create_lightgcn_ssl_embed()
        ssl_loss = self.calc_ssl_loss_v0(interaction, u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2)
        
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calcate sub loss
        # u_emb = u_g_embeddings_sub2[user]
        # p_i_emb = i_g_embeddings_sub2[pos_item]
        # n_i_emb = i_g_embeddings_sub2[neg_item]
        # p_s = torch.einsum("ij, ij->i", [u_emb, p_i_emb])
        # n_s = torch.einsum("ij, ij->i", [u_emb, n_i_emb])
        # mf_loss2 = self.mf_loss(p_s, n_s)

        # calculate reg Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        loss = mf_loss + self.reg_weight * reg_loss + self.ssl_reg * ssl_loss
        self.iter_step1 += 1
        self.tensorboard.add_scalar('mf_loss1', mf_loss.item(), self.iter_step1)
        # s,elf.tensorboard.add_scalar('mf_loss2', mf_loss2.item(), self.iter_step1)
        self.tensorboard.add_scalar('ssl_loss', ssl_loss.item(), self.iter_step1)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

