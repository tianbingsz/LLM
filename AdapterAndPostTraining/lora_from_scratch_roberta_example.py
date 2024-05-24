"""
build LORA (LoRA: Low-Rank Adaptation of Large Language Models) from scratch taking an example based on Roberta model.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

from torch.nn import MultiheadAttention
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention


class LORARobertaSelfAttention(RobertaSelfAttention):
    """A module, which inherits from the standard RobertaSelfAttention,
    but initializes and adds the LoRA matricies to it.
    rank: rank of the matrice B(d, rank), A(rank, d)
    alpha: \delta W = alpha BA
    """

    def __init__(self, rank=8, alpha=16, *args, **kwargs):
        super(LORARobertaSelfAttention, self).__init__(*args, **kwargs)
        d = self.all_head_size  # num_heads * head_size

        self.alpha = alpha
        # Initialize trainable matrices for query and value vectors
        self.lora_query_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_query_A = nn.Parameter(torch.randn(rank, d))
        self.lora_value_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_value_A = nn.Parameter(torch.randn(d, rank))

    """
        W^v x = W^v x + alpha B^v A^v x
    """

    def lora_query(self, x):
        lora_delta_weights = self.lora_query_B @ self.lora_query_A
        return self.query(x) + self.alpha * F.linear(x, lora_delta_weights)

    """
        W^q x = W^q x + alpha B^q A^q x
    """

    def lora_value(self, x):
        lora_delta_weights = self.lora_value_B @ self.lora_value_A
        return self.value(x) + self.alpha * F.linear(x, lora_delta_weights)

    """Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
        but replaced the query and value calls with calls to the lora_query and lora_value functions.
    """

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.tensor]:

        mixed_query_layer = self.lora_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.lora_value(encoder_hidden_states)
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class LORARoberta(nn.Module):
    def __init__(
        self,
        model_id="roberta-base",
        dropout_rate=0.1,
        lora_rank=8,
        lora_alpha=16,
        train_bias=True,
        train_embedding=False,
        train_layer_norm=True,
        num_classes=None,
    ):
        """
        Initializes a LoraWrapperRoberta instance, which is a wrapper around the RoBERTa model incorporating
        Low-Rank Adaptation (LoRA) to efficiently retrain the model for different NLP tasks such as GLUE benchmarks
        and SQuAD. LoRA allows for effective adaptation of large pre-trained models like RoBERTa with minimal updates.

        Parameters
        ----------
        num_classes : int, optional
            The number of classes for the classification layer on top of the RoBERTa model. The default value is
            determined by the task type if not provided. Has to be provided for the glue task.
        dropout_rate : float, default 0.1
            Dropout rate to be used in the dropout layers of the model.
        model_id : str, default "roberta-base"
            Identifier for the pre-trained RoBERTa model to be loaded.
        lora_rank : int, default 8
            Rank of the adaptation applied to the attention layers via LoRA.
        lora_alpha : int, default 16
            Regularization of the adaptation applied to the attention layers via LoRA.
        train_bias : bool, default True
            Flag indicating whether to update bias parameters during training.
        train_embedding : bool, default False
            Flag indicating whether to update embedding layer weights during training.
        train_layer_norms : bool, default True
            Flag indicating whether to update the layer norms during training. Usually this is a good idea.

        Examples
        --------
        To initialize a model for the 'glue' task type:

            model = LORARoberta()
        """
        super.__init__()

        self.dropout_rate = dropout_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.train_bias = train_bias
        self.train_embedding = train_embedding
        self.train_layer_norm = train_layer_norm

        # 1. Initialize the base model with parameters
        self.model_id = model_id
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_id)
        self.model = RobertaModel.from_pretrained(self.model_id)
        self.model_config = self.model.config

        # 2. Add the layer for the benchmark tasks
        # Get the output size of the base model & save other parameters
        d_model = self.model_config.hidden_size
        self.d_model = d_model
        self.num_classes = num_classes

        # Define the additional norm, linear layer, and dropout
        self.finetune_head_norm = nn.LayerNorm(d_model)
        self.finetune_head_dropout = nn.Dropout(dropout_rate)
        self.finetune_classifier = nn.Linear(d_model, num_classes)

        # 3. set up the lora model for training in Benchmark task:
        self.replace_multihead_attention(self.model)
        self.freeze_parameters_except_lora_and_bias()

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask)

        # Take the hidden states output from the base model
        x = outputs.last_hidden_state
        x = x[:, 0, :]  # Take output from [CLS] token, (batch_size, seq_len, dim)

        # add additonal layers for classfication
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_classifier(x)

        return x

    """
        Recursively replace new LORA injected attention layers with old self attention layers
    """

    def replace_multihead_attention(self, model):

        # Model can also be a module if it contains sub-components
        for name, subModule in model.named_children():
            if isinstance(subModule, LORARobertaSelfAttention):
                # replace with LORA self attention layer
                new_layer = LORARobertaSelfAttention(
                    rank=self.lora_rank, alpah=self.lora_alpha
                )
                old_state_dict = subModule.state_dict()

                # load state dict to new layer
                new_layer.load_state_dict(old_state_dict, strict=False)

                # replace new with old self attention layer
                setattr(model, name, new_layer)

            else:
                self.replace_multihead_attention(subModule)

    def freeze_parameters_except_lora_and_bias(self):
        """
        Freezes all parameters in the model, except those in LoRA layers, the finetune head, and bias parameters, if specified.
        All lora parameters are identified by having a name that starts with *lora_*.
        All finetune head parameters are identified by having a name that starts with *finetune_head_*.
        """
        for name, parameter in self.model.named_parameters():
            is_trainable = (
                "lora_" in name
                or "finetune_head_" in name
                or (self.train_bias and "bias" in name)
                or (self.train_embedding and "embeddings" in name)
                or (self.train_layer_norms and "LayerNorm" in name)
            )
            parameter.required_grad = is_trainable

    def save_lora_state_dict(self, lora_file_path):
        """
        Save the trainable parameters of the model, saves the state dict to that file
        """
        # fetch trainable parameters
        state_dict = {
            name: parameter
            for name, parameter in self.model.named_parameters
            if parameter.required_grad
        }

        # add addional parameters to state dict
        state_dict["model_id"] = self.model_id
        state_dict["lora_rank"] = self.lora_rank
        state_dict["lora_alpha"] = self.lora_alpha
        state_dict["num_classes"] = self.num_classes

        torch.save(state_dict, lora_file_path)

    @staticmethod
    def load_lora_state_dict(lora_file_path):
        """
        Load a state dict into the model from a specified file path.
        This is a staticmethod to be used from the base clase, returning a fully initialized and LoRA loaded model.

        Parameters
        ----------
        lora_file_path: the function will load the state dict from the file.

        Returns
        -------
        LORARoberta object, initialized and with the LoRA weights loaded.
        """
        # load parameters and states from file
        state_dict = torch.load(lora_file_path)
        instance = LORARoberta(
            model_id=state_dict["model_id"],
            lora_rank=state_dict["lora_rank"],
            lora_alpha=state_dict["lora_alpha"],
            num_classes=state_dict["num_classes"],
        )
        # copy state dict (from file) to new LORA Roberta Instance
        instance.load_state_dict(state_dict, strict=False)
        return instance
