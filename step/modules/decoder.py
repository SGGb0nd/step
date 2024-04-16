import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from step.manager import logger


class Exp(nn.Module):
    def forward(self, input):
        return torch.exp(input)


class ProbDecoder(nn.Module):
    """ProbDecoder is a probabilistic decoder to estimate the parameters of the output distribution(zinb or nb).

    Attributes:
        args (dict): The arguments to initialize the model.
    """

    def __init__(
        self,
        input_dim=3000,
        hidden_dim=128,
        n_hidden_layers=1,
        output_dim=64,
        norm="batch",
        dist="zinb",
        skip_dim=0,
        use_skip=False,
        num_batches: int = 1,
        use_l_scale: bool = False,
        activation: Literal["softplus", "softmax"] | None = None,
    ):
        """Initialize the ProbDecoder.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            n_hidden_layers (int): The number of hidden layers.
            output_dim (int): The output dimension.
            norm (str): The normalization method. Default is "batch".
            dist (str): The distribution of the output. Default is "zinb".
            skip_dim (int): The dimension of the skip connection. Default is 0.
            use_skip (bool): Whether to use skip connection. Default is False.
        """
        super(ProbDecoder, self).__init__()
        max_n_hidden_layers = math.log2(max(1, output_dim // hidden_dim)) + 1
        n_hidden_layers = min(n_hidden_layers, int(max_n_hidden_layers))
        self.args = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            dist=dist,
            norm=norm,
            skip_dim=skip_dim,
            use_skip=use_skip,
            use_l_scale=use_l_scale,
            num_batches=num_batches,
            activation=activation,
        )
        skip_dim = 0 if not use_skip else skip_dim

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _get_norm_layer(norm, hidden_dim),
            nn.ReLU(),
        )
        self.hid = nn.Identity()
        if n_hidden_layers > 1:
            layers = nn.Sequential(
                *[
                    layer
                    for i in range(1, n_hidden_layers)
                    for layer in [
                        nn.Linear(
                            hidden_dim * (2 ** (i - 1)) + skip_dim,
                            hidden_dim * (2**i),
                        )
                        if i == 1
                        else nn.Linear(
                            hidden_dim * (2 ** (i - 1)), hidden_dim * (2**i)
                        ),
                        _get_norm_layer(norm, hidden_dim * (2**i)),
                        nn.ReLU(),
                    ]
                ]
            )
            self.hid = nn.Sequential(
                layers,
            )

        self.use_skip = use_skip
        if self.use_skip:
            logger.info("Using skip connection between encoder and decoder")
            hidden_dim = hidden_dim * (2 ** max(1, n_hidden_layers - 1))
        else:
            hidden_dim = hidden_dim * (2 ** (n_hidden_layers - 1))
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            _get_activation(activation, norm),
        )
        self.px_dropout_decoder = (
            nn.Linear(
                hidden_dim, output_dim) if dist == "zinb" else (lambda _: None)
        )
        self.px_r = nn.Parameter(torch.randn(num_batches, output_dim).squeeze(0))

        if num_batches > 1:
            self.get_px_r = lambda batch_label: self.px_r[batch_label]
        else:
            self.get_px_r = lambda _: self.px_r

        if use_l_scale:
            self.l_scale = nn.Parameter(torch.randn(num_batches, 1).squeeze(0))
            if num_batches > 1:
                self.get_l_scale = lambda batch_label: 1 + F.sigmoid(
                    self.l_scale[batch_label]
                )
            else:
                self.get_l_scale = (lambda _: 1 + F.sigmoid(self.l_scale))
        else:
            self.get_l_scale = lambda _: 1

    def forward(self, z, library, batch_label=None, z_=None):
        """
        Forward pass of the ProbDecoder.

        Args:
            z (torch.Tensor): The input tensor.
            library (torch.Tensor): The library size.
            z_ (torch.Tensor): The input tensor of the skip connection. Default is None.

        Returns:
            torch.Tensor: The rate of the output distribution.
            torch.Tensor: The dropout of the output distribution.
            torch.Tensor: The scale of the output distribution.
        """
        px = self.ffn_(z, z_)
        px_scale = self.px_scale_decoder(px)
        px_rate = library * px_scale * self.get_l_scale(batch_label)
        px_dropout = self.px_dropout_decoder(px)
        return px_rate, px_dropout, px_scale

    def ffn_(self, z, z_=None):
        """Forward pass of the feedforward network.

        Args:
            z (torch.Tensor): The input tensor.
            z_ (torch.Tensor): The input tensor of the skip connection. Default is None.

        Returns:
            torch.Tensor: The output tensor.
        """
        px = self.ffn(z)
        if self.use_skip:
            assert z_ is not None
            px = torch.cat([px, z_], dim=-1)
        px = self.hid(px)
        return px

    def dropout_(self, z, z_=None):
        """
        Dropout logoit of the output distribution.

        Args:
            z (torch.Tensor): The input tensor.
            z_ (torch.Tensor): The input tensor of the skip connection. Default is None.

        Returns:
            torch.Tensor: The dropout logit of the output distribution.
        """
        px = self.ffn_(z, z_)
        px_dropout = self.px_dropout_decoder(px)
        return px_dropout

    def copy(self, with_param=True):
        """
        Copy the model.

        Args:
            with_param (bool): Whether to copy the parameters. Default is True.

        Returns:
            ProbDecoder: The copied model.
        """
        new_model = ProbDecoder(**self.args)
        if with_param:
            new_model.load_state_dict(self.state_dict())
        return new_model


def _get_norm_layer(norm, dim):
    if norm == "batch":
        norm_layer = nn.BatchNorm1d(dim, momentum=0.01, eps=0.001)
    elif norm == "layer":
        norm_layer = nn.LayerNorm(dim)
    else:
        norm_layer = nn.Identity()
    return norm_layer


def _get_activation(activation, norm):
    if activation == "softplus":
        return nn.Softplus()
    elif activation == "softmax":
        return nn.Softmax(dim=-1)
    else:
        if norm == "batch":
            return nn.Softmax(dim=-1)
        return nn.Softplus()
