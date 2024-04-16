import math
from itertools import chain
from typing import Optional

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from step.modules.gnn import GCN


class TopNAttention(nn.Module):
    """TopNAttention is a class that implements the top-n sparse attention mechanism.

    Attributes:
        top_n (int): The number of top values to select.
        num_heads (int): The number of attention heads.
        embed_dim (int): The dimension of the input embeddings.
        q (nn.Linear): The linear layer for the query.
        kv (nn.Linear): The linear layer for the key and value.
        out (nn.Linear): The linear layer for the output.
        act (Any): The activation function to use. Defaults to F.relu.
        T (float): The temperature parameter used in the calculation of the attention.
    """

    def __init__(
        self,
        top_n,
        num_heads,
        embed_dim,
        activation="relu",
    ):
        """Initialize the TopNAttention class.

        Args:
            top_n (int): The number of top values to select.
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimension of the input embeddings.
            activation (str, optional): The activation function to use. Defaults to 'relu'.
        """

        super(TopNAttention, self).__init__()
        self.top_n = top_n
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.q = nn.Linear(embed_dim, embed_dim * num_heads)
        self.kv = nn.Linear(embed_dim, embed_dim * num_heads)
        self.out = nn.Linear(embed_dim * num_heads, embed_dim)
        self.act = F.relu if activation == "relu" else F.sigmoid
        self.T = 1 if activation == "relu" else math.sqrt(embed_dim)

    def forward(self, q, k, v):
        b, _, _ = q.shape
        q = self.q(q).reshape(b, self.num_heads, -1, self.embed_dim)
        k, v = (
            self.kv(torch.cat([k, v], dim=1))
            .reshape(b, self.num_heads, -1, self.embed_dim)
            .chunk(2, dim=2)
        )
        attn = torch.einsum("bhnk, bhmk->bhnm", q, k) / (self.embed_dim**0.5)
        attn = self.act(attn / self.T)
        # make non-top-n attn to 0
        attn = attn.mean(dim=1).squeeze(1)
        return v, attn.unsqueeze(1)


class Mixer(nn.Module):
    """Mixer is a model that uses a transformer and sparse attention to predict the cell type distribution for each spot.

    Attributes:
        anchors (torch.Tensor): The anchor tensor.
        signatures (torch.Tensor): The signature tensor.
        hidden_dim (int): The hidden dimension.
        solver (str, optional): The solver type. Defaults to 'attn'.
        n_spots (int, optional): The number of spots. Defaults to None.
        smoother (nn.Module, optional): The smoother module. Defaults to None.
        use_smoother (bool, optional): Whether to use the smoother. Defaults to False.
        n_layers (int, optional): The number of layers. Defaults to 2.
        g (Any, optional): The spatial graph. Defaults to None.
        T (float, optional): The temperature parameter used in contrastive loss. Defaults to 0.07.
        max_ct_per_spot (int, optional): The maximum count per spot. Defaults to 8.
        hard_anchors (bool, optional): Whether to use hard anchors, otherwise use anchors adjusted by the attention and spot logits. Defaults to False.
        alpha (Any, optional): The alpha parameter used to control the sparsity of the attention. Defaults to None.
        rate (float, optional): The rate parameter used to scale the final loss. Defaults to 0.08.
        domain_wise (bool, optional): Whether to infer the cell type distribution domain-wise. Defaults to True.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        anchors,
        signatures,
        hidden_dim,
        solver="attn",
        n_spots=None,
        smoother=None,
        use_smoother=False,
        n_glayers=2,
        g=None,
        max_ct_per_spot: Optional[int] = 8,
        hard_anchors: Optional[bool] = False,
        domain_wise=True,
        **kwargs,
    ):
        """Initialize the Mixer class.

        Args:
            anchors (torch.Tensor): The anchor tensor.
            signatures (torch.Tensor): The signature tensor.
            hidden_dim (int): The hidden dimension.
            solver (str, optional): The solver type. Defaults to 'params'.
            n_spots (int, optional): The number of spots. Defaults to None.
            smoother (nn.Module, optional): The smoother module. Defaults to None.
            use_smoother (bool, optional): Whether to use the smoother. Defaults to True.
            n_layers (int, optional): The number of layers. Defaults to 2.
            g (Any, optional): The g parameter. Defaults to None.
            T (float, optional): The T parameter. Defaults to 0.07.
            max_ct_per_spot (int, optional): The maximum count per spot. Defaults to 8.
            hard_anchors (bool, optional): Whether to use hard anchors. Defaults to False.
            alpha (Any, optional): The alpha parameter used to control the sparsity of the attention. Defaults to None.
            rate (float, optional): The rate parameter used to scale the final loss. Defaults to 0.08.
            domain_wise (bool, optional): Whether to infer the cell type distribution domain-wise. Defaults to True.
            **kwargs: Additional keyword arguments.
        """

        super(Mixer, self).__init__()
        self.register_buffer("anchors", anchors)
        self.register_buffer("signatures", signatures)
        input_dim = anchors.shape[-1]
        if n_glayers is not None:
            self.smoother = (
                GCN(input_dim, input_dim, n_layers=n_glayers, with_edge=False)
                if smoother is None
                else smoother
            )
            self.g = g
            self.use_smoother = use_smoother
        else:
            self.smoother = nn.Identity()
            self.use_smoother = False

        self.ct_per_spot = max_ct_per_spot
        self.n_celltypes = anchors.shape[0]
        self.hard_anchors = hard_anchors
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.solver = solver
        self.decoder_layer = nn.Sequential(
            nn.TransformerEncoder(
                self.decoder_layer,
                num_layers=2,
            )
        )
        self.sc_consist = kwargs.get("sc_consist", True) and self.hard_anchors
        if hard_anchors:
            self.decoder_layer = nn.Identity()

        top_n = (
            max_ct_per_spot
            if max_ct_per_spot is not None and max_ct_per_spot < self.n_celltypes
            else self.n_celltypes
        )

        if self.solver == "params":
            assert n_spots is not None and n_spots > 0
            self.pz_param = nn.Parameter(
                torch.randn(n_spots, self.n_celltypes))
            self.pz = TopNAttention(
                top_n=top_n,
                num_heads=8,
                embed_dim=hidden_dim,
                activation="sigmoid",
            )
        else:
            self.pz = TopNAttention(
                top_n=top_n,
                num_heads=8,
                embed_dim=hidden_dim,
                activation="relu",
            )

        self.st_scale = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.ct_scale = nn.Parameter(torch.randn(1, self.n_celltypes))

        self.domain_wise = domain_wise
        self.hyperparams = kwargs
        self.T = kwargs.get("T", math.sqrt(input_dim))
        self.args = dict(
            hidden_dim=hidden_dim,
            solver=solver,
            n_spots=n_spots,
            smoother=smoother,
            use_smoother=use_smoother,
            n_glayers=n_glayers,
            max_ct_per_spot=max_ct_per_spot,
            hard_anchors=hard_anchors,
            domain_wise=domain_wise,
            hyperparams=kwargs,
        )

    @property
    def anchors_(self):
        return self.anchors

    def forward(self, h, g=None):
        """
        Forward pass for the Mixer model.

        Args:
            h (torch.Tensor): The input tensor.
            g (torch.Tensor, optional): The spatial graph. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The similarity matrix, spot logits, anchor logits, and top-n indices.
        """
        if hasattr(self, "g") or g is not None:
            g = self.g if g is None else g
            h = self.smoother(g, h)

        logits = h.unsqueeze(1)
        anchors = repeat(self.anchors_, "n d -> b n d", b=h.shape[0])
        logits = torch.cat([logits, anchors], dim=1)
        logits = self.decoder_layer(logits)
        spot_logits = logits[:, 0]
        anchors_logits = logits[:, 1:]
        pz = self.inf_pz(spot_logits, anchors_logits)
        if self.ct_per_spot is not None and self.ct_per_spot < self.n_celltypes:
            _, top_n_indices = torch.topk(
                pz, self.ct_per_spot, dim=-1, sorted=False)
        else:
            top_n_indices = repeat(
                torch.arange(self.n_celltypes), "n -> b n", b=h.shape[0]
            )

        return (
            pz,
            spot_logits,
            anchors_logits,
            top_n_indices,
        )

    def ts_forward(self, h, g=None, return_anchors=False):
        if hasattr(self, "g") or g is not None:
            g = self.g if g is None else g
            h = self.smoother(g, h)

        logits = h.unsqueeze(1)
        anchors = repeat(self.anchors_, "n d -> b n d", b=h.shape[0])
        logits = torch.cat([logits, anchors], dim=1)
        logits = self.decoder_layer(logits)
        spot_logits = logits[:, 0]
        anchors_logits = logits[:, 1:]
        if return_anchors:
            return anchors_logits
        return spot_logits

    def inf_pz(self, spot_logits, anchors_logits):
        """
        Calculate the inference similarity (pz) based on spot logits and anchors logits.

        Args:
            spot_logits (Tensor): Logits for spot.
            anchors_logits (Tensor): Logits for anchors.

        Returns:
            Tensor: Inference probability (pz).
        """
        spot_logits = spot_logits.unsqueeze(1)
        _, prop = self.pz(spot_logits, anchors_logits, anchors_logits)
        pz = prop.squeeze(1)
        if self.solver != "params":
            return pz
        return pz * F.softplus(self.pz_param)

    def get_prop(self, h, g=None, phat=True, ct_scale=True):
        """
        Calculate the probability distribution for the given input.

        Args:
            h (torch.Tensor): The input tensor.
            g (torch.Tensor, optional): The g tensor. Defaults to None.
            phat (bool, optional): Whether to calculate p_hat. Defaults to True.
            ct_scale (bool, optional): Whether to apply ct scaling. Defaults to True.

        Returns:
            torch.Tensor: The probability distribution.

        """
        if hasattr(self, "g") or g is not None:
            g = self.g if g is None else g
            h = self.smoother(g, h)

        logits = h.unsqueeze(1)
        anchors = repeat(self.anchors_, "n d -> b n d", b=h.shape[0])
        logits = torch.cat([logits, anchors], dim=1)
        logits = self.decoder_layer(logits)
        spot_logits = logits[:, 0]
        anchors_logits = logits[:, 1:]
        pz = self.inf_pz(spot_logits, anchors_logits)
        pz = self.mask_prop(pz)

        if not phat:
            return pz
        return self.p_hat(pz, spot_logits)

    def get_scores(self, h, g=None):
        """
        Calculate the scores between the input features and the anchor features.

        Args:
            h (torch.Tensor): The input features.
            g (torch.Tensor, optional): The anchor features. Defaults to None.

        Returns:
            torch.Tensor: The scores between the input features and the anchor features.
        """

        spot_anchors = self.ts_forward(h, g=g, return_anchors=True)
        spot_anchors = F.normalize(spot_anchors)
        anchors = F.normalize(self.anchors)
        return torch.einsum("nkd, kd -> nk", [spot_anchors, anchors])

    def mask_prop(self, prop):
        """
        Masks the given property tensor by selecting the top `ct_per_spot` values for each spot.

        Args:
            prop (torch.Tensor): The input property tensor.

        Returns:
            torch.Tensor: The masked property tensor.
        """
        if self.ct_per_spot is not None and self.ct_per_spot < self.n_celltypes:
            top_n_indices = torch.topk(prop, self.ct_per_spot, dim=-1)[1]
            top_n_prop = torch.zeros_like(prop)
            top_n_prop.scatter_(-1, top_n_indices,
                                prop.gather(-1, top_n_indices))
            return top_n_prop
        return prop

    def sc_consistency(self, top_n_anchors, anchors, indices):
        """
        Calculate the consistency loss for self-contrasting.

        Args:
            top_n_anchors (torch.Tensor): Tensor of shape (batch_size, num_anchors, embedding_dim) representing the top-n anchors.
            anchors (torch.Tensor): Tensor of shape (batch_size, num_anchors, embedding_dim) representing the anchors.
            indices (torch.Tensor): Tensor of shape (batch_size,) representing the indices of the positive anchors.

        Returns:
            torch.Tensor: The consistency loss.

        """
        logits = torch.einsum("bd,nd->bn", top_n_anchors, anchors)
        logits = logits / self.T
        labels = indices
        loss = F.cross_entropy(logits, labels)
        return loss

    def p_hat(self, coef, logits):
        ct_scale = self.ct_scale
        return ct_scale.exp() * coef  # * self.st_scale(logits).exp()

    def params(self):
        """
        Returns an iterator over the parameters of the model.

        Returns:
            iterator: An iterator over the parameters.
        """
        return chain(
            self.decoder_layer.parameters(),
            self.smoother.parameters(),
            self.pz.parameters(),
        )

    def phat_params(self):
        """
        Returns an iterator over the parameters of the `st_scale` and `ct_scale` models.
        """
        return chain(
            self.st_scale.parameters(),
            self.ct_scale,
        )
