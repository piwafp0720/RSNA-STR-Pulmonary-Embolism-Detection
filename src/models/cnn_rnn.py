import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .layers import pooling


def _set_model(model_name: str, 
               pretrained_path: str,
               model_param: dict):
    model = getattr(models, model_name)(**model_param)
    ckpt = torch.load(str(pretrained_path),
                    map_location='cpu')['state_dict']
    ckpt = {k[k.find('.') + 1:]: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    print(f'load checkpoint: {pretrained_path}')

    return model


class CNNEncoder(torch.nn.Module):
    def __init__(self, 
                 model_name: str, 
                 pretrained_path: str, 
                 model_param: dict):
        super(CNNEncoder, self).__init__()
        self.cnn_model = _set_model(
            model_name, pretrained_path, model_param)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor [batchsize=1, sequence, c, h, w]

        Returns
        -------
        output : torch.Tensor [batchsize=1, sequence, n_embeddings]
        """
        _, s, c, h, w = x.size()
        x = x.view(s, c, h, w)

        x = self.cnn_model.forward_until_pooling(x)
        s, f = x.size()
        x = x.view(1, s, f)

        return x


class RNNHead(torch.nn.Module):
    def __init__(self, 
                 model_name, 
                 pretrained_path: str,
                 model_param: dict,):
        super(RNNHead, self).__init__()
        self.rnn_model = _set_model(
            model_name, pretrained_path, model_param)

    def forward(self, *x):
        """
        Parameters
        ----------
        x : torch.Tensor [batchsize=1, sequence, n_embeddings]
            Image embeddings

        Returns
        -------
        image_level_output : torch.Tensor [batchsize=1, sequence]
            prediction result for "pe_present_on_image"
        exam_level_output: torch.Tensor [batchsize=1, 9]
            prediction result for exam-level labels
        """

        return self.rnn_model(*x)

class CNN_RNN(nn.Module):
    def __init__(self, 
                 cnn_model: str, 
                 cnn_pretrained_path: str,
                 cnn_param: dict,
                 rnn_model: str,
                 rnn_pretrained_path: str,
                 rnn_param: dict,):
        super(CNN_RNN, self).__init__()
        self.cnn = CNNEncoder(cnn_model,
                              cnn_pretrained_path,
                              cnn_param)

        self.rnn = RNNHead(rnn_model,
                           rnn_pretrained_path,
                           rnn_param,)

    def forward(self, x_3d):
        """
        Parameters
        ----------
        x_3d : torch.Tensor [batchsize, sequence, c, h, w]
               Sequence images

        Returns
        -------
        output : torch.Tensor [batchsize, sequence, n_embeddings]
            prediction result
        """
        return self.rnn(self.cnn(x_3d))