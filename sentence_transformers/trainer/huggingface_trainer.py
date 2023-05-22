""" A Trainer that is compatible with Huggingface transformers """
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy

from sentence_transformers import SentenceTransformer


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


class SentenceTransformerModel(nn.Module):
    def __init__(self, sentence_transformer, text_columns, scale: float = 20.0, similarity_fct=cos_sim):
        super(SentenceTransformerModel, self).__init__()
        self.sentence_transformer = sentence_transformer
        self.text_columns = text_columns
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        negative_input_ids=None,
        negative_attention_mask=None,
        labels=None,
    ):
        inputs = {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "negative_input_ids": negative_input_ids,
            "negative_attention_mask": negative_attention_mask,
        }
        features = self._collect_features(inputs)

        output = [self.sentence_transformer(sentence_feature)['sentence_embedding'] for sentence_feature in features]
        embeddings_a = output[0]
        embeddings_b = torch.cat(output[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, labels)

        output = torch.cat(output, dim=1)
        return loss, output

    def _collect_features(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        # SentenceTransformer model expects input_ids and attention_mask as input
        all_keys = [k.split(self.text_columns[0] + '_')[-1] for k in inputs.keys() if self.text_columns[0] in k]
        features = []
        for column in self.text_columns:
            one_set = {}
            for k in all_keys:
                one_set[k] = inputs[f"{column}_{k}"]
            features.append(one_set)
        return features
        # return [
        #     {
        #         "input_ids": inputs[f"{column}_input_ids"],
        #         "attention_mask": inputs[f"{column}_attention_mask"]
        #     }
        #     for column in self.text_columns
        # ]


@dataclass
class SentenceTransformersCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenizer: PreTrainedTokenizerBase
    text_columns: List[str]

    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, text_columns: List[str], max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # batch = {"label": torch.tensor([row["label"] for row in features])}
        batch = {}
        for column in self.text_columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=True,
            truncation='longest_first',
            return_tensors="pt",
            max_length=self.max_seq_length
        )
        # tokens = self.tokenizer(
        #     texts,
        #     return_attention_mask=True
        # )
        # return self.tokenizer.pad(
        #     tokens,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )


# Old version that doens't work
# class SentenceTransformersTrainer(Trainer):
#     def __init__(self, *args, text_columns: List[str], loss: nn.Module, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.text_columns = text_columns
#         self.loss = loss
#         self.loss.to(self.model.device)
#
#     def compute_loss(
#         self,
#         model: SentenceTransformer,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         return_outputs: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
#
#         features = self.collect_features(inputs)
#         loss = self.loss(features, inputs.get("label", None))
#         if return_outputs:
#             output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
#             return loss, output
#         return loss
#
#     def collect_features(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[Dict[str, torch.Tensor]]:
#         """Turn the inputs from the dataloader into the separate model inputs."""
#         # SentenceTransformer model expects input_ids and attention_mask as input
#         return [
#             {
#                 "input_ids": inputs[f"{column}_input_ids"],
#                 "attention_mask": inputs[f"{column}_attention_mask"]
#             }
#             for column in self.text_columns
#         ]
