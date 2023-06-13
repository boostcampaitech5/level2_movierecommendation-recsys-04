import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position."""

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            args.max_seq_length, args.hidden_size
        )

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(
            args.hidden_size / args.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_hidden_layers)]
        )

    def forward(
        self, hidden_states, attention_mask, output_all_encoded_layers=True
    ):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.attribute_embeddings = nn.Embedding(
            args.attribute_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            args.max_seq_length, args.hidden_size
        )
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(
        self, sequence_output, attribute_embedding
    ):
        """
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        """
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        """
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        sequence_output = self.mip_norm(
            sequence_output.view([-1, self.args.hidden_size])
        )  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(sequence_output, target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        """
        :param context: [B H]
        :param segment: [B H]
        :return:
        """
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    #
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(
        self,
        attributes,
        masked_item_sequence,
        pos_items,
        neg_items,
        masked_segment_sequence,
        pos_segment,
        neg_segment,
    ):
        # Encode masked sequence

        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(
            sequence_emb, sequence_mask, output_all_encoded_layers=True
        )
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        aap_loss = self.criterion(
            aap_score, attributes.view(-1, self.args.attribute_size).float()
        )
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * (
            masked_item_sequence != 0
        ).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(
            mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)
        )
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        map_loss = self.criterion(
            map_score, attributes.view(-1, self.args.attribute_size).float()
        )
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(
            segment_context, segment_mask, output_all_encoded_layers=True
        )

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(
            torch.unsqueeze(pos_segment_mask, 1), 1
        )
        pos_segment_encoded_layers = self.item_encoder(
            pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True
        )
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(
            torch.unsqueeze(neg_segment_mask, 1), 1
        )
        neg_segment_encoded_layers = self.item_encoder(
            neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True
        )
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(
            segment_context, pos_segment_emb
        )
        neg_segment_score = self.segment_prediction(
            segment_context, neg_segment_emb
        )

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(
            self.criterion(
                sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)
            )
        )

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(
            torch.ones(attn_shape), diagonal=1
        )  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.args.initializer_range
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BERT4RecModel(nn.Module):
    def __init__(self, args):
        super(BERT4RecModel, self).__init__()
