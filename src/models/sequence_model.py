import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import pooling


class DoubleLSTMModel(nn.Module):
    def __init__(self, 
                 lstm_depth: int, 
                 n_encoder_embeddings: int, 
                 n_lstm_hidden: int,
                 n_fc_hidden: int, 
                 n_classes: int, 
                 drop_rate: float, 
                 bidirectional:bool,
                 b_attention: bool,):
        super(DoubleLSTMModel, self).__init__()
        self.b_attention = b_attention
        self.LSTM_1 = nn.LSTM(
            input_size=n_encoder_embeddings,
            hidden_size=n_lstm_hidden,
            num_layers=lstm_depth,
            bias=False,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop_rate,
        )
        self.LSTM_2 = nn.LSTM(
            input_size=n_lstm_hidden * 2 if bidirectional else n_lstm_hidden, 
            hidden_size=n_lstm_hidden, 
            num_layers=lstm_depth,
            bias=False,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop_rate)

        self.image_level_f1 = nn.Linear(n_lstm_hidden * 2 
            if bidirectional else n_lstm_hidden, n_fc_hidden, bias=False)
        self.image_level_f2 = nn.Linear(n_fc_hidden, 1, bias=False)

        if self.b_attention:
            self.attention_layer_1 = nn.Linear(n_lstm_hidden * 2
                if bidirectional else n_lstm_hidden, 1)
            self.attention_layer_2 = nn.Linear(n_lstm_hidden * 2
                if bidirectional else n_lstm_hidden, 1)
        self.exam_level_f1 = nn.Linear(n_lstm_hidden * 2 
            if bidirectional else n_lstm_hidden, n_fc_hidden, bias=False)
        self.exam_level_f2 = nn.Linear(n_fc_hidden, n_classes - 1, bias=False)

        self.r = nn.ReLU()
        self.d = nn.Dropout(drop_rate)
    
    def forward(self, x, sequence_length=None):
        if sequence_length is not None:
            return self.forward_with_masking(x, sequence_length)
        else:
            return self.forward_simple(x)

    def forward_simple(self, x):
        """
        called when input is mini-batch of uniform length samples 
        or single sample (i.e. batch_size=1)

        Parameters
        ----------
        x : torch.Tensor [batchsize, sequence, n_embeddings]
            Image embeddings

        Returns
        -------
        image_level_output : torch.Tensor [batchsize=1, sequence]
            prediction result for "pe_present_on_image"
        exam_level_output: torch.Tensor [batchsize=1, 9]
            prediction result for exam-level labels
        """
        self.LSTM_1.flatten_parameters()
        self.LSTM_2.flatten_parameters()
        h_lstm1, (hn, hc) = self.LSTM_1(x)
        h_lstm2, (hn, hc) = self.LSTM_2(h_lstm1)
        h = h_lstm1 + h_lstm2

        image_level_output = self.image_level_f2(
            self.d(self.r(self.image_level_f1(h)))).squeeze(-1)

        if self.b_attention:
            attention_w1 = F.softmax(self.attention_layer_1(h_lstm1), dim=1)
            attention_w2 = F.softmax(self.attention_layer_2(h_lstm2), dim=1)
            h_lstm1 = torch.sum(attention_w1 * h_lstm1, dim=1)
            h_lstm2 = torch.sum(attention_w2 * h_lstm2, dim=1)
            h = h_lstm1 + h_lstm2
        else:
            h_lstm1 = torch.sum(h_lstm1, dim=1)
            h_lstm2 = torch.sum(h_lstm2, dim=1)
            h = h_lstm1 + h_lstm2

        exam_level_output = self.exam_level_f2(
            self.d(self.r(self.exam_level_f1(h))))
        
        return image_level_output, exam_level_output

    def forward_with_masking(self, x, sequence_length):
        """
        called when input is mini-batch of variable length samples.
        process ignoring padded part.

        Parameters
        ----------
        x : torch.Tensor [batchsize, max_sequence, n_embeddings]
            Image embeddings

        Returns
        -------
        image_level_output : torch.Tensor [batchsize, max_sequence]
            prediction result for "pe_present_on_image"
        exam_level_output: torch.Tensor [batchsize, 9]
            prediction result for exam-level labels
        """
        self.LSTM_1.flatten_parameters()
        self.LSTM_2.flatten_parameters()
        h_lstm1, (hn, hc) = self.LSTM_1(x)
        h_lstm2, (hn, hc) = self.LSTM_2(h_lstm1)
        h = h_lstm1 + h_lstm2

        image_level_output = self.image_level_f2(
            self.d(self.r(self.image_level_f1(h)))).squeeze(-1)

        hidden = []
        for h1, h2, n_sequence in zip(h_lstm1, h_lstm2, sequence_length):
            h1 = h1[:n_sequence, :]
            h2 = h2[:n_sequence, :]
            if self.b_attention:
                attention_w1 = F.softmax(self.attention_layer_1(h1), dim=0)
                attention_w2 = F.softmax(self.attention_layer_2(h2), dim=0)
                h1 = torch.sum(attention_w1 * h1, dim=0, keepdim=True)
                h2 = torch.sum(attention_w2 * h2, dim=0, keepdim=True)
                h = h1 + h2
            else:
                h1 = torch.sum(h1, dim=0, keepdim=True)
                h2 = torch.sum(h2, dim=0, keepdim=True)
                h = h1 + h2
            hidden.append(h)
        hidden = torch.cat(hidden)

        exam_level_output = self.exam_level_f2(
            self.d(self.r(self.exam_level_f1(hidden))))
        
        return image_level_output, exam_level_output


class SequenceModel(nn.Module):
    def __init__(self,
                 base,
                 lstm_depth: int,
                 n_embeddings: int, 
                 n_lstm_hidden: int,
                 n_fc_hidden: int,
                 n_classes: int, 
                 drop_rate: float,
                 b_bidirectional: bool,
                 b_attention: bool = True,
                 b_stack_diff: bool = False,):
        super(SequenceModel, self).__init__()
        self.b_stack_diff = b_stack_diff
        n_embeddings = 3 * n_embeddings if self.b_stack_diff else n_embeddings
        self.rnn = DoubleLSTMModel(
            lstm_depth,
            n_embeddings,
            n_lstm_hidden,
            n_fc_hidden,
            n_classes,
            drop_rate,
            b_bidirectional,
            b_attention,)


    def forward(self, *args):

        return self.rnn(*args)
        
