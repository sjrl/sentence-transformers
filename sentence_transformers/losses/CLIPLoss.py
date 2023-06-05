import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util


class CLIPLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the XX.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structring the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim, symmetric_loss: bool = False):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(CLIPLoss, self).__init__()
        self.model = model
        # self.scale = nn.Parameter(torch.tensor(scale_init), requires_grad=True)
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.symmetric_loss = symmetric_loss

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        if self.symmetric_loss:
            loss = -1. * (torch.softmax(scores, dim=1).diag().mean() + torch.softmax(scores, dim=0).diag().mean()) / 2.0
        else:
            loss = -1. * torch.softmax(scores, dim=1).diag().mean()
        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


# # Suggestion from https://discuss.pytorch.org/t/regarding-clamped-learnable-parameter/58474/3
# class Clamp(torch.autograd.Function):
#     # clamp_class = Clamp()
#     # self.z = nn.Parameter(torch.tensor(1.0), requires_grad=True)
#     # clamp_class.apply(self.z)
#     @staticmethod
#     def forward(ctx, input):
#         return input.clamp(min=0, max=1) # the value in iterative = 2
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clone()
