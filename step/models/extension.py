from itertools import chain
from typing import Literal, Optional

import torch
from einops import repeat
from torch import nn

from step.models.transcriptformer import TranscriptFormer


class BatchAwareScale(nn.Module):
    """BatchAwareScale is a module that performs BatchAwareScale introduced in the paper.

    Attributes:
        net (nn.Sequential): The neural network.
        act (nn.Module): The activation function.
    """

    def __init__(self, input_dim, output_dim, act=None):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            batch (Tensor): The batch data.

        Returns:
            Tensor: The output data after scaling.
        """
        super(BatchAwareScale, self).__init__()
        input_dim = input_dim + output_dim
        hidden_dim = input_dim * 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.act = nn.ReLU() if act == "relu" else nn.Identity()

    def forward(self, x, batch):
        scale = self.net(torch.cat([batch, x], dim=1))
        return self.act(x * scale)


class BatchAwareLayerNorm(nn.Module):
    """BatchAwareLayerNorm is a module that performs BatchAwareLayerNorm introduced in the paper.

    Attributes:
        mean (nn.Linear): The mean layer.
        scale (nn.Linear): The scale layer.
        layernorm (nn.LayerNorm): The layer normalization layer.
        act (nn.Module): The activation function.
    """

    def __init__(self, input_dim, output_dim, act="relu"):
        """
        Initialize the BatchAwareLayerNorm module.

        Args:
            input_dim (int): The input dimension of the module.
            output_dim (int): The output dimension of the module.
            act (str, optional): The activation function to use. Defaults to 'relu'.
        """
        super(BatchAwareLayerNorm, self).__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.scale = nn.Linear(input_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.act = nn.ReLU() if act == "relu" else nn.Identity()

    def forward(self, x, batch):
        x = self.layernorm(x)
        mean = self.mean(batch)
        scale = self.scale(batch).exp()
        return self.act(x * scale + mean)


class NrmlsBC(TranscriptFormer):
    """NrmlsBC is an extension of the TranscriptFormer model that supports batch-aware normalization and scaling to eliminate batch effects.

    Attributes:
        num_batches (int): The number of batches.
        batch_emb_dim (int): The batch embedding dimension.
        smoother (Optional[nn.Module]): The smoother module.
        batch_embedding (nn.Parameter): The batch embedding parameter.
        moduler (TranscriptFormer): The moduler module.
        batch_readout (BatchAwareScale): The batch readout module.
        args (Dict[str, Any]): The arguments of the model.
    """

    def __init__(self,
                 num_batches: int,
                 num_classes: int = 1,
                 dispersion: Literal['gene', 'batch-gene'] = 'batch-gene',
                 use_l_scale=False,
                 **kwargs):
        """Initialize the Extension class.

        Args:
            num_batches (int): The number of batches.
            num_classes (int, optional): The number of classes. Defaults to 1.
            **kwargs: Additional keyword arguments.

        """

        module_dim = kwargs["module_dim"]
        batch_emb_dim = kwargs.pop("batch_emb_dim", module_dim)

        num_model_batches = num_batches if dispersion == 'batch-gene' else 1
        super().__init__(num_batches=num_model_batches, use_l_scale=use_l_scale, **kwargs)
        self.num_batches = num_batches
        self.batch_emb_dim = batch_emb_dim
        self.batch_embedding = nn.Parameter(
            torch.randn(num_batches, self.batch_emb_dim)
        )  # type:ignore
        self.moduler.layernorm = BatchAwareLayerNorm(  # type:ignore
            input_dim=self.batch_emb_dim,
            output_dim=self.input_dim,
        )
        self.batch_readout = BatchAwareScale(
            input_dim=self.batch_emb_dim,
            output_dim=module_dim,
        )

        self.args["dispersion"] = dispersion
        self.args["num_batches"] = num_batches
        self.args["num_classes"] = num_classes

    def _tsfmr_forward(self, x, batch_rep):
        """
        Forward pass of the _tsfmr_forward method.

        Args:
            x: Input tensor.
            batch_rep: Batch representation tensor.

        Returns:
            rep: Output representation tensor.
        """
        batch_rep = batch_rep @ self.batch_embedding
        auto_fold = self.moduler(x, batch_rep)
        b, _, _ = auto_fold.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        auto_fold = torch.cat([cls_tokens, auto_fold], dim=1)  # type:ignore
        auto_fold = self.expand(auto_fold)
        rep = self.module(auto_fold)
        return rep

    def encode_ts(self, x, batch_rep):
        """Encodes the output of transformer encoders using the transformer model.

        Args:
            x (Tensor): The output of transformer encoders.
            batch_rep (bool): Whether to return the representation for each time step in the batch.

        Returns:
            Tensor: The encoded representation of the output of transformer encoders.
        """
        rep = self._tsfmr_forward(x, batch_rep)
        return rep[:, 0]

    def readout_(self, tsfmr_out):
        """Apply smoothing to the transformer output if enabled, and then perform readout.

        Args:
            tsfmr_out: The transformer output.

        Returns:
            The result of the readout operation.
        """
        if self._smooth:
            tsfmr_out = self.local_smooth(tsfmr_out)
        return self.readout(tsfmr_out)

    def readout_batch(self, rep_ts, batch_rep):
        """Readout the representation with batch representation.

        Args:
            rep_ts: The representation tensor.
            batch_rep: The batch representation.

        Returns:
            The class representation.
        """
        cls_rep = self.readout_(rep_ts)
        batch_rep = batch_rep @ self.batch_embedding
        cls_rep = self.batch_readout(cls_rep, batch_rep, relu=False)
        return cls_rep

    def encode(self, x, batch_rep):
        """Encodes the input data `x` using the specified batch representation `batch_rep`.

        Args:
            x: The input data to be encoded.
            batch_rep: The batch representation to be used for encoding.

        Returns:
            The encoded representation of the input data.
        """
        cls_rep = self.encode_ts(x, batch_rep)
        cls_rep = self.readout_(cls_rep)
        return cls_rep

    def decode(self, cls_rep, x_gd, batch_rep, rep_ts=None):
        """Decodes the given input representation into output representation.

        Args:
            cls_rep: The class representation.
            x_gd: The input representation.
            batch_rep: The batch representation (optional).

        Returns:
            The decoded output representation.
        """
        if batch_rep is None:
            return super().decode(cls_rep, x_gd)
        return self.decode_skip(
            cls_rep=cls_rep,
            rep_ts=rep_ts,
            x_gd=x_gd,
            batch_rep=batch_rep,
        )

    def decode_ts(self, rep_ts, x_gd, batch_rep=None):
        """Decode the given representation tensor `rep_ts` into a prediction tensor.

        Args:
            rep_ts (torch.Tensor): The representation tensor.
            x_gd (torch.Tensor): The input tensor.
            batch_rep (torch.Tensor, optional): The batch representation tensor. Defaults to None.

         Returns:
            torch.Tensor: The prediction tensor.
        """
        if batch_rep is None:
            return super().decode_ts(rep_ts, x_gd)
        cls_rep = self.readout_(rep_ts)
        return self.decode_skip(cls_rep, rep_ts, x_gd, batch_rep)

    def decode_skip(self, cls_rep, rep_ts, x_gd, batch_rep):
        """Decodes the input data using the skip model.

        Args:
            cls_rep: The class representation.
            rep_ts: The representation time series.
            x_gd: The input data.
            batch_rep: The batch representation.

        Returns:
            A dictionary containing the decoded values:
            - px_rate: The rate of the decoded values.
            - px_dropout: The dropout of the decoded values.
            - px_scale: The scale of the decoded values.
            - px_r: The px_r value.
            - decoder_type: The type of decoder.
            - x: The input data.
        """

        px_rate, px_dropout, px_scale = None, None, None
        batch_label = batch_rep.argmax(dim=1)
        batch_rep = batch_rep @ self.batch_embedding
        cls_rep = self.batch_readout(cls_rep, batch_rep)
        library = x_gd.sum(-1, keepdim=True)
        if cls_rep.shape[0] > library.shape[0]:
            num_repeat = cls_rep.shape[0] // library.shape[0]
            library.unsqueeze_(1)
            library = repeat(library, "b () d -> b n d", n=num_repeat)
            library = library.reshape(-1, 1)
        px_rate, px_dropout, px_scale = self.decoder(
            cls_rep,
            library,
            batch_label=batch_label,
            z_=rep_ts,
        )
        return dict(
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_scale=px_scale,
            px_r=self.get_px_r(batch_label),
            decoder_type=self.decoder_type,
            x=x_gd,
        )

    def decode_(self, cls_rep, x_gd, batch_rep=None):
        return super().decode(cls_rep, x_gd)

    def forward(self, x, batch_rep, return_exp=True):
        h = self.encode(x, batch_rep)
        (
            x_recon,
            _,
        ) = self.decode(h, batch_rep=batch_rep, x_gd=x)
        return x_recon

    def init_anchor(self, num_classes: Optional[int] = None, new_anchors=True):
        """Initializes the anchor module.

        Args:
            num_classes (Optional[int]): The number of classes. If provided and greater than 0,
                the class classification head and anchors will be initialized accordingly.
            new_anchors (bool): Whether to initialize new anchors.

        Returns:
            bool: True if the anchor module is successfully initialized, False otherwise.
        """
        if (num_classes is not None) and num_classes > 0:
            self.num_classes = num_classes
            self.class_clsf_head = nn.Sequential(
                nn.LayerNorm(self.module_dim),
                nn.Linear(self.module_dim, self.module_dim),
                nn.Linear(self.module_dim, num_classes),
            )
            if new_anchors:
                self.anchors = nn.Parameter(
                    torch.randn(num_classes, self.module_dim)
                )  # type:ignore
            return True
        return False

    def copy(self, with_state=True):
        """Creates a copy of the current object.

            Returns:
            A new instance of the NrmlsBC class with the same arguments and state.
        """
        copied_model = NrmlsBC(**self.args)
        if with_state:
            state_dict = self.state_dict()
            if 'anchors' in state_dict:
                self.init_anchor(state_dict['anchors'].shape[0])
            # if hasattr(self, 'smoother'):
            #     self.init_smoother_with_builtin()
            copied_model.load_state_dict(state_dict)
        return copied_model

    def copy_dec(self):
        """Creates a copy of the model with the specified parameters.

        Returns:
            nn.ModuleDict: A copy of the model with the registered parameters.
        """
        model = nn.ModuleDict(
            {
                "readout": self.readout,
                "batch_readout": self.batch_readout,
                "decoder": self.decoder,
            }
        )
        model.register_parameter("batch_embedding", self.batch_embedding)
        model.register_parameter("px_r", self.px_r)
        return model

    @property
    def non_decoder_param(self):
        """Returns an iterator over the non-decoder parameters of the model.

        This includes the parameters of the `moduler`, `expand`, `module`, and `readout` modules.
        """
        return chain(
            self.moduler.parameters(),
            self.expand.parameters(),
            self.module.parameters(),
            self.readout.parameters(),
        )
