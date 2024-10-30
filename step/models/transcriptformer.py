from typing import Literal, Optional

import dgl
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from step.manager import logger
from step.modules.decoder import ProbDecoder
from step.modules.gnn import GCN
from step.modules.transformer import Transformer

drop_edge = dgl.DropEdge(p=0.5)


class Linear2D(nn.Module):
    """Linear2D module consists of a linear layer with 3D weight matrix.

    Args:
        input_dim (int): The input dimension of the Linear2D module.
        hidden_dim (int): The hidden dimension of the Linear2D module.
        n_modules (int): The number of modules of the Linear2D module.
        bias (bool, optional): Whether to use bias. Defaults to False.
    """

    def __init__(self, input_dim, hidden_dim, n_modules, bias=False):
        """Linear2D module consists of a linear layer with 3D weight matrix.

        Args:
            input_dim (int): dimension of input
            hidden_dim (int): dimension of hidden layer
            n_modules (int): number of linear modules
            bias (bool, optional): whether to use bias. Defaults to False.
        """
        super(Linear2D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modules = n_modules

        self.weights = torch.randn(input_dim, hidden_dim, n_modules)
        self.weights = nn.Parameter(
            nn.init.xavier_normal_(self.weights))  # type:ignore
        self.bias = None
        if bias:
            self.bias = torch.randn(1, hidden_dim, n_modules)
            self.bias = nn.Parameter(
                nn.init.xavier_normal_(self.bias))  # type:ignore

    def __repr__(self):
        return f"Linear2D(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, n_modules={self.n_modules})"

    def forward(self, x):
        affine_out = torch.einsum("bi,ijk->bjk", [x, self.weights])
        if self.bias is not None:
            affine_out = affine_out + self.bias
        return affine_out


class Readout(nn.Module):
    """Readout module for the TranscriptFormer model.

    Attributes:
        net (nn.Sequential): The sequential neural network.
        variational (bool): Whether to use variational encoding.
        out (nn.Sequential): The sequential neural network for the output.
        mean (nn.Linear): The linear layer for the mean.
        logvar (nn.Linear): The linear layer for the logvar.

    """

    def __init__(self, input_dim, output_dim, variational=True):
        """Initializes the Readout module.

        Args:
            input_dim (int): The input dimension of the Readout module.
            output_dim (int): The output dimension of the Readout module.
            variational (bool, optional): Whether to use variational encoding. Defaults to True.
        """
        super(Readout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(input_dim),
        )
        self.variational = variational
        if not variational:
            self.out = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.mean = nn.Linear(input_dim, output_dim)
            self.logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Readout module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.net(x)
        if not self.variational:
            out = self.out(h)
            return out
        mean = self.mean(h)
        logvar = self.logvar(h)
        setattr(self, "mu", mean)
        setattr(self, "sigma", logvar.exp().sqrt())
        setattr(self, "dist", Normal(self.mu, self.sigma))
        return self.dist.rsample()

    def kl_loss(self):
        """
        Computes the KL divergence loss.

        Returns:
            torch.Tensor: The KL divergence loss.
        """
        mean = torch.zeros_like(self.mu)
        scale = torch.ones_like(self.sigma)
        if not hasattr(self, "dist"):
            return 0
        kl_loss = kl(self.dist, Normal(mean, scale))
        return kl_loss

    def clear(self):
        if not self.variational:
            return
        for attr in ["mu", "sigma", "dist"]:
            if hasattr(self, attr):
                delattr(self, attr)


class GeneModuler(nn.Module):
    """GeneModuler takes gene expression as input and outputs gene modules.

    Attributes:
        input_dim (int): The input dimension of the GeneModuler model.
        hidden_dim (int): The hidden dimension of the GeneModuler model.
        n_modules (int): The number of modules of the GeneModuler model.
        layernorm (nn.LayerNorm): The layer normalization layer.
        extractor (Linear2D): The Linear2D object.
    """

    def __init__(self, input_dim=2000, hidden_dim=8, n_modules=16):
        """GeneModuler takes gene expression as input and outputs gene modules.

        Args:
            input_dim (int, optional): dimension of input. Defaults to 2000.
            hidden_dim (int, optional): dimension of hidden layer. Defaults to 8.
            n_modules (int, optional): number of modules. Defaults to 16.
        """
        super(GeneModuler, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modules = n_modules

        self.layernorm = nn.LayerNorm(input_dim)
        self.extractor = Linear2D(
            input_dim=input_dim, hidden_dim=hidden_dim, n_modules=n_modules
        )

    def forward(self, x, batch=None):
        # x = x + self.gene_tokens
        if batch is not None:
            module = self.layernorm(x, batch)
        else:
            module = self.layernorm(x)
        module = self.extractor(x).transpose(2, 1)
        return F.relu(module)

    def demodule(self, x):
        unfolded = self.extractor.transpose(x)
        return unfolded

    def random_permute(self, x):
        perm = torch.randperm(self.input_dim)
        return x[:, perm]


class TranscriptFormer(nn.Module):
    """TranscriptFormer is a gene expression model based on the Transformer architecture.

    Attributes:
        input_dim (int): The input dimension of the TranscriptFormer model.
        module_dim (int): The module dimension of the TranscriptFormer model.
        hidden_dim (int): The hidden dimension of the TranscriptFormer model.
        n_modules (int): The number of modules of the TranscriptFormer model.
        moduler (GeneModuler): The GeneModuler object.
        expand (nn.Linear): The linear layer for expanding the module.
        readout (Readout): The Readout object.
        module (nn.TransformerEncoder): The TransformerEncoder object.
        cls_token (nn.Parameter): The classification token.
        px_r (torch.nn.Parameter): The parameter for the zero-inflated negative binomial distribution.
        decoder (ProbDecoder): The ProbDecoder object.
        decoder_type (str): The type of the decoder.
        _smooth (bool): Whether to use smoothing.
        smoother (GCN): The GCN object for smoothing.
        smoother_type (str): The type of the smoother.
        args (dict): The arguments for the TranscriptFormer model.
        gargs (dict): The arguments for the GCN object.
    """

    def __init__(
        self,
        decoder_type="zinb",
        use_pe=True,
        use_smooth=False,
        use_skip=False,
        input_dim=2000,
        module_dim=30,
        decoder_input_dim=None,
        hidden_dim=256,
        n_modules=16,
        nhead=8,
        n_enc_layer=3,
        dec_norm="batch",
        variational=True,
        smoother="GCN",
        n_glayers=3,
        dec_hidden_dim=None,
        n_dec_hid_layers: int = 1,
        edge_clip=2,
        use_l_scale: bool = False,
        num_batches: int = 1,
        activation: Literal["softplus", "softmax"] | None = None,
    ):
        """Initializes the TranscriptFormer model.

        Args:
            grids (None, optional): Grids. Defaults to None.
            decoder_type (str, optional): Decoder type. Defaults to 'zinb'.
            use_pe (bool, optional): Whether to use positional encoding. Defaults to True.
            use_smooth (bool, optional): Whether to use smoothing. Defaults to False.
            use_skip (bool, optional): Whether to use skip connections. Defaults to False.
            input_dim (int, optional): Input dimension. Defaults to 2000.
            module_dim (int, optional): Module dimension. Defaults to 30.
            decoder_input_dim (None, optional): Decoder input dimension. Defaults to None.
            hidden_dim (int, optional): Hidden dimension. Defaults to 256.
            n_modules (int, optional): Number of modules. Defaults to 16.
            nhead (int, optional): Number of attention heads. Defaults to 8.
            n_enc_layer (int, optional): Number of encoder layers. Defaults to 3.
            dec_norm (str, optional): Decoder normalization. Defaults to 'batch'.
            variational (bool, optional): Whether to use variational encoding. Defaults to True.
            smoother (str, optional): Smoother type. Defaults to 'GCN'.
            n_glayers (int, optional): Number of graph layers. Defaults to 3.
            dec_hidden_dim (None, optional): Decoder hidden dimension. Defaults to None.
            n_dec_hid_layers (int, optional): Number of decoder hidden layers. Defaults to 1.
            edge_clip (int, optional): Edge clip value. Defaults to 2.
        """
        super(TranscriptFormer, self).__init__()
        if not variational:
            logger.info("Not using VAE")
        self.input_dim = input_dim
        self.module_dim = module_dim
        self.hidden_dim = hidden_dim
        self.n_modules = n_modules
        if decoder_input_dim is None:
            decoder_input_dim = module_dim

        self.moduler = GeneModuler(
            input_dim=input_dim, hidden_dim=module_dim, n_modules=n_modules
        )
        self.expand = (
            nn.Linear(module_dim, hidden_dim)
            if module_dim != hidden_dim
            else nn.Identity()
        )
        self.readout = Readout(hidden_dim, module_dim, variational=variational)
        if use_pe:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=4 * hidden_dim,
                batch_first=True,
            )
            self.module = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_enc_layer,
            )
        else:
            self.module = Transformer(
                dim=hidden_dim,
                depth=n_enc_layer,
                heads=nhead,
                dim_head=hidden_dim,
                mlp_dim=4 * hidden_dim,
            )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, module_dim))  # type:ignore
        assert decoder_type in ["nb", "zinb", "poisson"]
        dec_hidden_dim = hidden_dim if dec_hidden_dim is None else dec_hidden_dim

        # if n_glayers is not None:
        #     self.smoother = GCN(
        #         in_feats=hidden_dim,
        #         h_feats=hidden_dim,
        #         with_edge=False,
        #         n_layers=n_glayers,
        #     )

        self.decoder = ProbDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=dec_hidden_dim,
            output_dim=input_dim,
            use_skip=use_skip,
            skip_dim=hidden_dim,
            n_hidden_layers=n_dec_hid_layers,
            norm=dec_norm,
            dist=decoder_type,
            use_l_scale=use_l_scale,
            num_batches=num_batches,
            activation=activation,
        )
        self.decoder_type = decoder_type
        self._smooth = use_smooth

        self.smoother_type = smoother

        self.args = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            module_dim=module_dim,
            n_modules=n_modules,
            decoder_type=decoder_type,
            variational=variational,
            edge_clip=edge_clip,
            use_pe=use_pe,
            n_dec_hid_layers=n_dec_hid_layers,
            dec_norm=dec_norm,
            decoder_input_dim=decoder_input_dim,
            use_skip=use_skip,
            n_glayers=n_glayers,
        )
        self.gargs = dict(
            in_feats=hidden_dim,
            h_feats=hidden_dim,
            with_edge=False,
            n_layers=n_glayers,
        )

    def get_px_r(self, batch_label):
        return self.decoder.get_px_r(batch_label)

    def init_smoother_with_builtin(self):
        self.smoother = GCN(**self.gargs)
        return True

    def local_smooth(self, h, g: Optional[dgl.DGLGraph] = None):
        """
        Local smoothing function.

        Args:
            h (torch.Tensor): The input tensor.
            g (Optional[dgl.DGLGraph], optional): The graph. Defaults to None.
        """
        if g is None:
            g = self.g
        return self.smoother(g, h)

    def encode_ts(self, x, batch_rep=None) -> torch.Tensor:
        """
        Encode the input tensor with only the transformer.

        Args:
            x (torch.Tensor): The input tensor.
            batch_rep ([type], optional): representation tensor of the batch indicator. Defaults to None.

        Returns:
            torch.Tensor: The encoded tensor, denoted as non-standardized representation.
        """
        auto_fold = self.moduler(x)
        b, _, _ = auto_fold.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        auto_fold = torch.cat([cls_tokens, auto_fold], dim=1)  # type:ignore
        auto_fold = self.expand(auto_fold)
        rep = self.module(auto_fold)
        cls_rep = rep[:, 0]
        return cls_rep

    def readout_(self, cls_rep) -> torch.Tensor:
        """
        Readout function.

        Args:
            cls_rep (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self._smooth:
            cls_rep = self.local_smooth(cls_rep)
        return self.readout(cls_rep)

    def encode(self, x, batch_rep=None) -> torch.Tensor:
        """
        Encode the input tensor with the transformer and the readout function.

        Args:
            x (torch.Tensor): The input tensor.
            batch_rep ([type], optional): representation tensor of the batch indicator. Defaults to None.

        Returns:
            torch.Tensor: The encoded tensor, denoted as standardized representation.
        """
        cls_rep = self.encode_ts(x)
        cls_rep = self.readout_(cls_rep)
        return cls_rep

    def decode_ts(self, rep_ts, x_gd, batch_rep=None):
        """
        Decoding process starting from the non-standardized representation.

        Args:
            rep_ts (torch.Tensor): The input tensor.
            x_gd (torch.Tensor): The input tensor.
            batch_rep ([type], optional): representation tensor of the batch indicator. Defaults to None.
        """
        px_rate, px_dropout, px_scale = None, None, None
        library = x_gd.sum(-1).unsqueeze(-1)
        cls_rep = self.readout_(rep_ts)

        if cls_rep.shape[0] > library.shape[0]:
            num_repeat = cls_rep.shape[0] // library.shape[0]
            library.unsqueeze_(1)
            library = repeat(library, "b () d -> b n d", n=num_repeat)
            library = library.reshape(-1, 1)
        if self.decoder_type != "transformer":
            px_rate, px_dropout, px_scale = self.decoder(
                cls_rep, library, z_=rep_ts)
        else:
            px_rate, px_dropout = self.decoder(cls_rep)
        return dict(
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_scale=px_scale,
            px_r=self.get_px_r(batch_rep),
            decoder_type=self.decoder_type,
            x=x_gd,
        )

    def decode(self, cls_rep, x_gd, batch_rep=None):
        px_rate, px_dropout, px_scale = None, None, None
        library = x_gd.sum(-1).unsqueeze(-1)
        if cls_rep.shape[0] > library.shape[0]:
            num_repeat = cls_rep.shape[0] // library.shape[0]
            library.unsqueeze_(1)
            library = repeat(library, "b () d -> b n d", n=num_repeat)
            library = library.reshape(-1, 1)
        px_rate, px_dropout, px_scale = self.decoder(cls_rep, library)
        return dict(
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_scale=px_scale,
            px_r=self.get_px_r(batch_rep),
            decoder_type=self.decoder_type,
            x=x_gd,
        )

    def blocks_forward(self, blocks, h):
        h = self.smoother.batch_forward(blocks, h)
        return self.readout(h)

    def forward(self, x):
        h = self.encode(x)
        if self.decoder_type in ["zinb", "nb"]:
            x_recon, _, _ = self.decoder(h, x)
        else:
            x_recon, _ = self.decoder(h, x)
        return x_recon

    def copy(self, with_state=True):
        new_model = TranscriptFormer(**self.args)
        if with_state:
            new_model.load_state_dict(self.state_dict())
        return new_model
