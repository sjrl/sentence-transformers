""" A Trainer that is compatible with Huggingface transformers """
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy

from sentence_transformers import SentenceTransformer


class SentenceTransformerModel(nn.Module):
    def __init__(self, sentence_transformer, loss):
        super(SentenceTransformerModel, self).__init__()
        self.sentence_transformer = sentence_transformer
        self.loss_model = loss

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
        loss = self.loss(features, labels)
        output = torch.cat([self.sentence_transformer(row)["sentence_embedding"][:, None] for row in features], dim=1)
        return loss, output

    def _collect_features(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        # SentenceTransformer model expects input_ids and attention_mask as input
        return [
            {
                "input_ids": inputs[f"{column}_input_ids"],
                "attention_mask": inputs[f"{column}_attention_mask"]
            }
            for column in self.text_columns
        ]


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

    def __init__(self, tokenizer: PreTrainedTokenizerBase, text_columns: List[str]) -> None:
        self.tokenizer = tokenizer
        self.text_columns = text_columns

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # batch = {"label": torch.tensor([row["label"] for row in features])}
        batch = {}
        for column in self.text_columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(texts, return_attention_mask=False)
        return self.tokenizer.pad(
            tokens,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )


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
