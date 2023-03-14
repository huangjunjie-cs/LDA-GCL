import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, normal_, constant_


def kaiming_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        kaiming_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def normal_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding, std=0.1):
        normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        normal_(module.weight.data)



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


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2).pow(2)
            else:
                reg_loss = reg_loss + W.norm(2).pow(2)
        return reg_loss

