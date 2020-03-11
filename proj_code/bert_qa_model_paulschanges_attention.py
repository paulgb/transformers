#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:46:32 2020

@author: raghuramkowdeed
"""
import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class BertForQuestionAnswering2(BertPreTrainedModel):
    def __init__(self, config, bert_hidden_states = 4 ,num_heads =1, dropout = 0.1):
        config = deepcopy(config)
        config.output_hidden_states = True
        print('conf', config)
        super(BertForQuestionAnswering2, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.bert_hidden_states = bert_hidden_states

        self.bert = BertModel(config)
        #config.num_labels = 1
        num_labels = 2
        
        self.qa_attn_project = nn.Linear(2*config.hidden_size*self.bert_hidden_states, num_labels)
        self.qa_attn = nn.MultiheadAttention(config.hidden_size*self.bert_hidden_states, 
                                             num_heads=num_heads, dropout = dropout)
        
        self.sm = nn.Softmax(dim=-1)

        self.init_weights()

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """
        #print('calling new bert qa model 3')
        [sequence_output, pooled_output, hidden_states] = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #print('hidden states:', len(hidden_states))
        
        #print(len( outputs[0]), outputs[0].shape, len(outputs[1]), outputs[1].shape, len(outputs[2]), len(outputs[2][0]), outputs[2][0][0].shape )

        #print('id', id(hidden_states[0]), id(sequence_output))
        #print('shapes:', [a.shape for a in hidden_states[1:(self.bert_hidden_states+1)]])
        sequence_output = (sequence_output,) + hidden_states[1:(self.bert_hidden_states)]
        #print('h0', [a.shape for a in sequence_output])
        sequence_output = torch.cat(sequence_output, dim=2)
        #print('h1', sequence_output.shape)
        
        #q_mask is used to mask words from paragraph and use words from question compute attention.
        # context_mask = attention_mask * token_type_ids
        q_mask = ( attention_mask == 0 ) | (token_type_ids == 1)
        #print('q', q_mask.shape)
        #q_mask[:,0] = False 
        
        #print('s0', sequence_output.shape)
        sequence_output_attn = sequence_output.permute(1,0,2)
        #print('s1', sequence_output.shape)

        
        sequence_output_attn, _ = self.qa_attn(sequence_output_attn, sequence_output_attn,
                                               sequence_output_attn, key_padding_mask=q_mask)
        #print('s2', sequence_output.shape)
        
        sequence_output_attn = sequence_output_attn.permute(1,0,2)
        #print('s3', sequence_output.shape)
        
        sequence_output = torch.cat((sequence_output, sequence_output_attn), 2)
        #print('s4', sequence_output.shape)
        
        #logits = self.qa_outputs1(sequence_output)
        logits = self.qa_attn_project(sequence_output)
        #print('ss', logits.shape)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits, hidden_states)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
