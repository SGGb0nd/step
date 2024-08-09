from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from einops import repeat
from torch import nn
from torch.utils.data import default_collate

from step.manager import logger
from step.modules.decoder import ProbDecoder
from step.utils.dataset import CrossDataset
from step.utils.gbolt import MultiGraphsAllNodesSampler
from step.utils.misc import aver_items_by_ct, generate_adj

from ..models.deconv import Mixer
from ..models.knn_map import knn_random_coords
from ..models.nnls import nnls_deconv
from .sc_funcmodel import scMultiBatchNrmls


class CrossModalityNrmls(scMultiBatchNrmls):
    """CrossModalityNrmls is a class for training and integrating cross-modality data: scRNA-seq and spatial transcriptomics.

    Attributes:
        need_anchor (bool): Whether to use anchors for integration.
        mixer (Mixer): The mixer module for integration.
        single_st (bool): Whether the dataset contains only single section.
        st_decoder (StDecoder): The self
        max_neighs (int): The maximum number of neighbors.
        edge_clip (Optional[float]): The edge clipping value.
        _num_batches (int): The number of batches.
        _gener_graph (Callable): The function for generating the graph.
        _rout (nn.Module): The readout module.
    """

    def __init__(self, device=None, **kwargs):
        """Initialize the CrossModalityNrmls.

        Args:
            need_anchor (bool, optional): Whether to use anchors for integration. Defaults to False.
            max_neighbors (int, optional): The maximum number of neighbors. Defaults to 10.
            edge_clip (Optional[float], optional): The edge clipping value. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.need_anchor = kwargs.pop("need_anchor", False)
        kwargs["use_skip"] = kwargs.get("use_skip", False)
        kwargs["dec_norm"] = kwargs.get("dec_norm", None)
        super().__init__(device=device, **kwargs)
        self._trained = False
        self.mixer: Mixer
        self.single_st: bool
        self.max_neighs = kwargs.pop("max_neighbors", 10)
        self.edge_clip = kwargs.get("edge_clip", None)
        self._num_batches = self.model.num_batches
        self._gener_graph = partial(
            generate_adj, edge_clip=self.edge_clip, max_neighbors=self.max_neighs
        )
        self._get_px_r: Callable
        self._rout: nn.Module

    def _init_mixer(
        self,
        anchors,
        signatures,
        hidden_dim,
        solver="attn",
        n_spots=Optional[int],
        max_ct_per_spot=None,
        hard_anchors=False,
        alpha=None,
        n_glayers=3,
        **kwargs,
    ):
        """
        Initializes the mixer module.

        Args:
            anchors (list): List of anchor points.
            signatures (list): List of signatures.
            hidden_dim (int): Dimension of the hidden layer.
            solver (str, optional): Solver type. Defaults to 'attn'.
            n_spots (int, optional): Number of spots. Defaults to None.
            max_ct_per_spot (int, optional): Maximum count per spot. Defaults to None.
            hard_anchors (bool, optional): Whether to use hard anchors. Defaults to False.
            alpha (float, optional): Alpha value. Defaults to None.
            n_glayers (int, optional): Number of graph layers. Defaults to 3.
            T (float, optional): Temperature value. Defaults to 0.07.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        if hidden_dim == self.model.module_dim:
            setattr(self, "_rout", nn.Identity().to(self.device))
        else:
            setattr(self, "_rout", self.model.readout)

        self.mixer = Mixer(
            anchors=anchors,
            signatures=signatures,
            solver=solver,
            n_spots=n_spots,
            smoother=None,
            g=getattr(self.model, "g", None),
            hidden_dim=hidden_dim,
            max_ct_per_spot=max_ct_per_spot,
            hard_anchors=hard_anchors,
            alpha=alpha,
            n_glayers=n_glayers,
            **kwargs,
        )

    def _init_st_decoder(
        self,
        n_dec_layers=1,
        decoder_type="zinb",
        single_st=True,
        dec_norm="batch",
        use_st_decoder=False,
        st_batches=None,
        activation: Literal['softmax', 'softplus'] = 'softplus',
        use_l_scale=True,
    ):
        """
        Initialize the st specific decoder.

        Args:
            n_dec_layers (int): Number of hidden layers in the decoder.
            decoder_type (str): Type of decoder distribution.
            single_st (bool): Whether the dataset contains only single section.
            dec_norm (str): Normalization method for the decoder.
            use_st_decoder (bool): Whether to use a separate st specific decoder.

        Returns:
            None
        """

        if use_st_decoder:
            args: dict = dict(**self.model.decoder.args)  # type:ignore
            args["use_skip"] = False
            args["n_hidden_layers"] = n_dec_layers
            args["norm"] = dec_norm
            args["dist"] = decoder_type
            args["activation"] = activation
            args["use_l_scale"] = use_l_scale
            args["num_batches"] = len(st_batches) if st_batches is not None else 1
            self.st_decoder = ProbDecoder(**args,)
            setattr(self, "st_decoder_type", decoder_type)
            setattr(self, "single_st", single_st)
            self.model.args["st_decoder"] = {  # type:ignore
                "n_dec_layers": n_dec_layers,
                "decoder_type": decoder_type,
                "single_st": single_st,
                "dec_norm": dec_norm,
                "use_st_decoder": use_st_decoder,
                "st_batches": st_batches,
                "activation": activation,
                "use_l_scale": use_l_scale,
            }
        else:
            self.st_decoder = self.model.decoder
            setattr(self, "st_decoder_type", self.model.decoder.args["dist"])
            setattr(self, "single_st", single_st)
        self._get_px_r = self.model.get_px_r

    def _st_decode(self, cls_rep, x_gd, batch_rep):
        """
        Decode the input data using the self-trans decoder.

        Args:
            cls_rep (torch.Tensor): The cell representation.
            x_gd (torch.Tensor): The input data.
            batch_rep (torch.Tensor): The batch representation.

        Returns:
            dict: A dictionary containing the decoded values.
                - px_rate (torch.Tensor): The rate parameter.
                - px_dropout (torch.Tensor): The dropout parameter.
                - px_scale (torch.Tensor): The scale parameter.
                - px_r (float): The r parameter.
                - decoder_type (str): The type of decoder used.
                - x (torch.Tensor): The input data.
        """
        batch_label = batch_rep.argmax(1)
        batch_rep = batch_rep @ self.model.batch_embedding
        cls_rep = self.model.batch_readout(cls_rep, batch_rep)
        px_rate, px_dropout = None, None
        library = x_gd.sum(-1, keepdim=True)
        if cls_rep.shape[0] > library.shape[0]:
            num_repeat = cls_rep.shape[0] // library.shape[0]
            library.unsqueeze_(1)
            library = repeat(library, "b () d -> b n d", n=num_repeat)
            library = library.reshape(-1, 1)

        px_rate, px_dropout, _ = self.st_decoder(cls_rep, library, batch_label)
        return dict(
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_r=self._get_px_r(batch_label),
            decoder_type=self.st_decoder_type,
            x=x_gd,
        )

    def _loss_st(self, input_tuple, smooth=True):
        """Calculate the loss for the self-training process.

        Args:
            input_tuple (tuple): A tuple containing the input tensors (x_gd, rep_ts, batch_label).
            smooth (bool, optional): Whether to apply local smoothing. Defaults to True.

        Returns:
            dict: A dictionary containing the loss values.
        """
        x_gd, rep_ts, batch_label = input_tuple
        x_gd = x_gd.to(self.device)
        rep_ts = rep_ts.to(self.device)
        batch_rep = self.get_batch_ohenc(batch_label).to(self.device)

        cls_k = self.model.readout(rep_ts)

        self.model._smooth = smooth
        cls_rep = self.model.readout_(rep_ts)

        decode_dict = self._st_decode(cls_rep, x_gd, batch_rep)
        loss_dict = self.loss_fn(**decode_dict)

        if smooth:
            loss_dict["contrast_loss"] = _contrastive_loss(cls_rep, cls_k)

        return loss_dict

    def _loss_st_dec(self, input_tuple):
        return self._loss_st(input_tuple, smooth=False)

    def _loss_mixer(self, input_tuple):
        """Calculates the loss for the mixer model.

        Args:
            input_tuple (tuple): A tuple containing the input tensors:
                x_gd (torch.Tensor): The input tensor.
                rep_ts (torch.Tensor): The representation tensor.
                domains (torch.Tensor): The domain tensor.
                batch_label (torch.Tensor): The batch label tensor.

        Returns:
            dict: A dictionary containing the loss values.
        """
        x_gd, rep_ts, domains, batch_label = input_tuple
        x_gd = x_gd.to(self.device)
        rep_ts = rep_ts.to(self.device)
        batch_rep = (
            self.
            get_batch_ohenc(batch_label, device=self.device)
        )
        domains = domains.long().to(self.device)

        rate1 = self.mixer.hyperparams.get("rate1", 1.0)
        rate2 = self.mixer.hyperparams.get("rate2", 1.0)

        coef, logits, tuned_anchors, indicies = self.mixer(rep_ts)
        tuned_anchors_ = self._rout(tuned_anchors)
        mix_rep = torch.einsum(
            "bn, bnc -> bc",
            coef,
            tuned_anchors_,
        )

        mix_decode_dict = self._st_decode(mix_rep, x_gd, batch_rep)
        px_dropout = None
        if mix_decode_dict["px_dropout"] is not None:
            mix_decode_dict["px_dropout"] = self.st_decoder.dropout_(
                self._rout(rep_ts))
            px_dropout = mix_decode_dict["px_dropout"]

        loss_dict = self.loss_fn(**mix_decode_dict, nokl=True)

        predicted_P = self.mixer.p_hat(coef, logits)
        library = x_gd.sum(-1, keepdim=True)
        px_scale = predicted_P @ self.mixer.signatures
        px_rate = px_scale * library
        decode_dict = dict(
            px_rate=px_rate,
            px_dropout=px_dropout,
            px_r=self._get_px_r(batch_label),
            decoder_type=self.st_decoder_type,
            x=x_gd,
        )
        loss_ori = self.loss_fn(nokl=True, **decode_dict,)["recon_loss"]
        loss_dict["loss_ori"] = rate1 * loss_ori
        loss_dict["recon_loss"] = rate2 * loss_dict["recon_loss"]

        if not self.mixer.sc_consist:
            loss_dict["sc_consistency"] = self.mixer.sc_consistency(
                tuned_anchors_.reshape(-1, mix_rep.shape[-1]),
                self._rout(
                    self.mixer.anchors,
                ),
                indicies.reshape(-1).to(self.device),
            )

        if self.mixer.domain_wise:
            intra_div = torch.tensor(0.).to(self.device)
            unique_domains = domains.unique()
            means = torch.zeros(
                len(unique_domains), coef.shape[1], device=self.device
            )

            coef = F.normalize(coef, p=2, dim=-1)
            for domain in unique_domains:
                _coef = coef[domains == domain]
                diff = _coef.unsqueeze(1) - _coef.unsqueeze(0)
                diff = (diff ** 2).sum(-1)
                intra_div += diff.sum() / (_coef.size(0) ** 2)
                means[domain] = _coef.mean(0)
            loss_dict["intra_div"] = intra_div

            inter_diff = means.unsqueeze(1) - means.unsqueeze(0)
            inter_diff = (inter_diff ** 2).sum(-1)
            inter_diff.fill_diagonal_(0)
            loss_dict["inter_div"] = -inter_diff.sum() / (unique_domains.numel() ** 2)

        alpha = self.mixer.hyperparams.get("alpha", None)
        if alpha is not None:
            loss_dict["l1_reg"] = alpha * torch.norm(coef, p=1)
        return loss_dict

    def handle_ginput_tuple(self, input_tuple, ind=None, step=None):
        g = input_tuple
        g = g.to(self.device)
        x_gd = g.ndata["feat"]
        batch_label = g.ndata["batch_label"]
        rep_ts = g.ndata["rep"]
        self.model.g = g
        loss_dict = self._loss_st((x_gd, rep_ts, batch_label))
        loss_dict["graph_ids"] = torch.unique(
            batch_label).cpu().numpy().tolist()
        self.model.g.to("cpu")
        return loss_dict

    def from_spatial_graph(
        self,
        adata: AnnData,
        dataset,
        batch_size: int = 1024,
        num_iters=1000,
        use_rep=True,
        batch_key: Optional[str] = None,
        **kwargs,
    ):
        """Convert spatial graph data into a dataloader for training.

        Args:
            adata (AnnData): Annotated data object containing spatial graph information.
            dataset (StDataset): Spatial transcriptomics dataset object.
            batch_size (int, optional): Number of samples per batch. Defaults to 1024.
            num_iters (int, optional): Number of iterations. Defaults to 1000.
            use_rep (bool, optional): Whether to use the representation data. Defaults to False.
            batch_key (str, optional): Key for batch information in `adata.obs`. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dataloader: A dataloader object for training.

        """
        graphs = []
        if batch_key is None:
            batch_key = dataset.batch_key
        for batch in adata.obs[batch_key].cat.categories:
            _adata = adata[adata.obs[batch_key] == batch]
            g = self._gener_graph(_adata)
            g.ndata["feat"] = dataset.get(batch, "gene_expr")
            if use_rep:
                if getattr(dataset, "rep", None) is not None:
                    g.ndata["rep"] = dataset.get(batch, "rep")
            g.ndata["batch_label"] = dataset.get(batch, "batch_label")
            graphs.append(g)
        batched_graph = dgl.batch(graphs)

        sampler = MultiGraphsAllNodesSampler(
            mode="node", budget=batch_size, n_graphs=batch_size
        )
        dataloader = dgl.dataloading.DataLoader(
            batched_graph,
            torch.arange(num_iters),
            sampler,
            persistent_workers=True,
            num_workers=1,
        )
        return dataloader

    @torch.no_grad()
    def add_embed_st(self, dataset, st_dataset=None, key_added="X_smoothed", readout=False):
        """Add the smoothed representation to the spatial transcriptomics dataset.

        Args:
            adata (AnnData): The annotated data object.
            dataset (CrossDataset): The cross dataset object.
            key_added (str, optional): The key to store the smoothed representation. Defaults to 'X_smoothed'.
        """

        if not dataset._integrated:
            dataset.set_rep(self.embed(dataset, tsfmr_out=True, as_numpy=False))
        st_dataset = dataset.subset_st() if st_dataset is None else st_dataset
        _adata = st_dataset.adata
        self.model.eval()
        self.st_decoder.eval()
        self.model.to(self.device)
        self.model.g = self._gener_graph(
            adata=_adata, batch_key=st_dataset.batch_key
        ).to(self.device)
        rep = self.model.smoother(
            self.model.g, st_dataset.rep.to(self.device)
        ).cpu()
        dataset.st_adata.obsm["X_rep_tsfmr"] = st_dataset.rep.cpu().numpy()
        if key_added is not None:
            if readout:
                rep = self.model.readout(rep.to(self.device)).cpu()
            dataset.st_adata.obsm[key_added] = rep.numpy()

    def integrate(
        self,
        adata: AnnData,
        dataset: CrossDataset,
        epochs=1,
        batch_size: Optional[int] = None,
        lr=1e-3,
        split_rate=0.2,
        tune_epochs=20,
        tune_lr=1e-4,
        need_anchors=True,
        key_added="X_rep",
        beta1=1e-2,
        beta2=1e-3,
        reset=False,
        kl_cutoff=None,
    ):
        """Integrate the input data using the cross-function model.

        Args:
            adata (AnnData): The input AnnData object.
            dataset (CrossDataset): The CrossDataset object containing the dataset information.
            epochs (int): The number of training epochs (default: 1).
            batch_size (Optional[int]): The batch size for training (default: None).
            lr (float): The learning rate for training (default: 1e-3).
            split_rate (float): The rate for splitting the data into training and validation sets (default: 0.2).
            tune_epochs (int): The number of epochs for fine-tuning (default: 20).
            tune_lr (float): The learning rate for fine-tuning (default: 1e-4).
            need_anchors (bool): Whether to use anchors for integration (default: True).
            anchors_only (bool): Whether to use only anchors for integration (default: False).
            key_added (str): The key to store the integrated representation in the AnnData object (default: 'X_rep').
            beta1 (float): The value of beta1 for the cross-function model (default: 1e-2).
            beta2 (float): The value of beta2 for the cross-function model (default: 1e-3).
        """

        super().run(
            adata=adata,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            split_rate=split_rate,
            tune_epochs=tune_epochs,
            tune_lr=tune_lr,
            need_anchors=need_anchors,
            unlabeled_key=dataset.st_sample_list,
            groupby=dataset.batch_key,
            key_added=key_added,
            beta1=beta1,
            beta2=beta2,
            reset=reset,
            kl_cutoff=kl_cutoff,
        )
        if need_anchors:
            mask = adata.obs[dataset.batch_key].isin(dataset.st_sample_list)
            dataset.st_adata.obsm["X_anchord"] = adata.obsm["X_anchord"][mask.values]

    def generate_domains(
        self,
        adata: AnnData,
        dataset: CrossDataset,
        decoder_type="zinb",
        use_rep=None,
        split_rate=0.2,
        epochs: int = 1,
        smooth_epochs=200,
        batch_size: int = 1,
        beta: Optional[float] = None,
        n_dec_layers: int = 2,
        lr=1e-3,
        tune_lr: Optional[float] = None,
        n_glayers=2,
        dec_norm="batch",
        key_added: str = "X_smoothed",
        use_st_decoder=False,
        graph_batch_size=2,
        kl_cutoff=None,
        logging=False,
        **kwargs,
    ):
        """Generates domains for the given dataset using the specified parameters.

        Args:
            adata (AnnData): The annotated data object.
            dataset (CrossDataset): The cross dataset object.
            decoder_type (str): The type of decoder to use. Default is 'zinb'.
            use_rep (Optional[str]): The representation to use. Default is None.
            split_rate (float): The split rate for train-test split. Default is 0.2.
            epochs (int): The number of epochs for training. Default is 1.
            smooth_epochs (int): The number of epochs for smoothing. Default is 200.
            batch_size (int): The batch size for training. Default is 1.
            beta (Optional[float]): The beta value for training. Default is None.
            n_dec_layers (int): The number of decoder layers. Default is 2.
            lr (float): The learning rate for training. Default is 1e-3.
            tune_lr (Optional[float]): The learning rate for fine-tuning. Default is None.
            n_glayers (int): The number of graph layers. Default is 2.
            dec_norm (str): The normalization method for decoder. Default is 'batch'.
            key_added (str): The key to add to the dataset. Default is 'X_smoothed'.
            use_st_decoder (bool): Whether to use the ST decoder. Default is False.
            logging (bool): Whether to enable logging. Default is False.
        """

        self.set_kl_cutoff(kl_cutoff)
        self.split_rate = split_rate
        writer = None
        if logging:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                comment=f"-gener-domains-{dataset.st_sample_names}")

        if not dataset._integrated:
            if use_rep is not None and use_rep in adata.obsm_keys():
                tsfmr_out = torch.from_numpy(adata.obsm[use_rep])
                assert tsfmr_out.shape[1] == self.model.hidden_dim
                tsfmr_out = adata.obsm[use_rep]
                tsfmr_out = torch.from_numpy(tsfmr_out)
            else:
                tsfmr_out = self.embed(dataset, tsfmr_out=True, as_numpy=False)
            dataset.set_rep(tsfmr_out)
        else:
            tsfmr_out = dataset.rep
        single_st = dataset.num_st_batches == 1
        st_dataset = dataset.subset_st()
        logger.info(st_dataset)

        _adata = st_dataset.adata

        st_batches = None if self.model.args['dispersion'] == 'gene' \
            else dataset.st_batch_codes.to_list()
        self._init_st_decoder(
            use_st_decoder=use_st_decoder,
            n_dec_layers=n_dec_layers,
            decoder_type=decoder_type,
            single_st=single_st,
            dec_norm=dec_norm,
            st_batches=st_batches,
            **kwargs,
        )

        self.model.to(self.device)
        self.st_decoder.to(self.device)
        if use_st_decoder:
            self.optimizer = torch.optim.Adam(
                self.st_decoder.parameters(), lr=lr
            )
            loaders = self.make_loaders(
                st_dataset,
                batch_size,
                split_rate=split_rate,
            )
            self.train_batch(
                epochs=epochs,
                loaders=loaders,
                call_func=self._loss_st_dec,
                writer=writer,
            )  # type:ignore
            logger.info("StDecoder trained")

        if beta is not None:
            self.set_beta(beta)

        self.st_decoder.eval()
        self.model.eval()
        self.model.init_smoother(n_glayers=n_glayers)
        self.model.smoother.train()
        self.model.smoother.to(self.device)
        self.use_earlystop = False
        self.model._smooth = True
        self.optimizer = torch.optim.Adam(self.model.smoother.parameters(), lr=lr)
        if tune_lr is not None:
            self.st_decoder.train()
            self.optimizer.add_param_group(
                {
                    "params": self.st_decoder.parameters(),
                    "lr": tune_lr,
                },
            )
        logger.info("Start training smoother")
        if self.single_st:
            self.split_rate = 0.0
            self.model.g = self._gener_graph(
                adata=_adata, batch_key=st_dataset.batch_key
            ).to(self.device)
            loaders = self.make_loaders(
                st_dataset, len(st_dataset), split_rate=0.0, shuffle=False
            )
            self.train_batch(
                epochs=smooth_epochs,
                loaders=loaders,
                call_func=self._loss_st,
                writer=writer,
            )  # type:ignore
        else:
            self.split_rate = 0.0
            gloader = self.from_spatial_graph(
                adata=_adata,
                dataset=st_dataset,
                batch_size=graph_batch_size,
                num_iters=smooth_epochs,
                use_rep=True,
            )
            self.train_node_sampler(
                epochs=1,  # type:ignore
                gloader=gloader,
                writer=None,
            )

        self.add_embed_st(dataset, st_dataset, key_added=key_added)
        self.model.g.to("cpu")
        self.model.cpu()

    def deconv(
        self,
        adata: AnnData,
        dataset: CrossDataset,
        cell_type_key: Optional[str] = None,
        st_sample_list: Optional[list] = None,
        domain_wise: bool = True,
        domain_key: str | None = "domain",
        library_size: float = 1e5,
        solver="attn",
        rep_key: Optional[str] = None,
        epochs: int = 1,
        batch_size: int | None = None,
        lr: float = 1e-3,
        max_ct_per_spot: Optional[int] = None,
        n_glayers: int | None = None,
        hard_anchors: bool = False,
        sc_consist: bool = True,
        alpha: float | None = None,
        use_raw: bool | Literal['st'] = 'st',
        batch_used: str | int | None = 0,
        logging=False,
        **kwargs,
    ):
        """Perform cell-type deconvolution on spatial transcriptomics data.

        Args:
            adata (AnnData): Annotated data matrix.
            dataset (CrossDataset): CrossDataset object containing spatial transcriptomics data.
            cell_type_key (str, optional): Key for cell type annotation in the dataset. Default is None.
            domain_wise (bool, optional): Whether to perform domain-wise deconvolution. Default is True.
            domain_key (str, optional): Key for domain information in the dataset. Default is 'domain'.
            library_size (int, optional): Library size. Default is 1e5.
            solver (str, optional): Solver type. Default is 'attn'.
            rep_key (Optional[str]): The key for the representation. Default is None.
            epochs (int, optional): Number of training epochs. Default is 1.
            batch_size (Optional[int]): The batch size for training. Default is None.
            lr (float, optional): Learning rate. Default is 1e-3.
            max_ct_per_spot (Optional[int]): Maximum count per spot. Default is None.
            n_glayers (Optional[int]): Number of graph layers. Default is None.
            hard_anchors (bool, optional): Whether to use hard anchors. Default is False.
            sc_consist (bool, optional): Whether to use self-consistency. Default is True.
            alpha (bool, optional): Alpha value. Default is None.
            use_raw (bool, optional): Whether to use raw data. Default is True.
            batch_used (str, int, optional): The batch used for deconvolution. Default is 0.
            T (float, optional): Temperature value. Default is 0.07.
            logging (bool, optional): Whether to enable logging. Default is False.
        """

        writer = None
        if logging:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                comment=f"-deconv-{dataset.st_sample_names}")

        st_adata = dataset.st_adata
        rep = self._assert_rep(adata, dataset, rep_key)

        if domain_key is not None:
            assert st_adata.obs.get(domain_key) is not None, f"{domain_key} not found in adata.obs"
            domains = pd.Categorical(st_adata.obs[domain_key]).codes
            logger.info(f"Domains: {list(st_adata.obs[domain_key].unique())}")
            logger.info(f"Domain Codes: {list(np.unique(domains))}")
            domains = torch.from_numpy(domains)
        else:
            domains = None
        st_dataset = dataset.subset_st(
            domains=domains,
        )
        if st_sample_list is not None:
            st_dataset.subset(
                key=st_sample_list,
                col=dataset.st_batch_key,
                exclude=False,
            )
        assert st_dataset.rep is not None

        single_st = dataset.num_st_batches == 1
        if not hasattr(self, "st_decoder"):
            self._init_st_decoder(
                use_st_decoder=False,
                single_st=single_st,
            )

        _get_px_r, st_decoder, st_decoder_type = self._set_st_counts(adata, dataset, library_size, use_raw, batch_used)

        if cell_type_key is None:
            cell_type_key = dataset.class_key

        signatures, batch_used_code, celltypes = self._get_ct_sig(adata, dataset, cell_type_key, use_raw, batch_used)

        self._init_mixer(
            anchors=aver_items_by_ct(
                dataset.adata, cell_type_key, rep, None),
            signatures=signatures,
            max_ct_per_spot=max_ct_per_spot,
            hidden_dim=rep.shape[-1],
            solver=solver,
            n_spots=st_adata.n_obs,
            hard_anchors=hard_anchors,
            alpha=alpha,
            n_glayers=n_glayers,
            domain_wise=domain_wise,
            sc_consist=sc_consist,
            **kwargs,
        )

        self.optimizer = torch.optim.Adam(self.mixer.parameters(), lr=lr)

        self._fix_params()

        batched = batch_size is not None and n_glayers is None
        loaders = self.make_loaders(
            st_dataset,
            batch_size=batch_size if batched else len(st_dataset),
            split_rate=0.0,
            shuffle=n_glayers is None,
            collate_fn=self._make_collate_fn(
                batch_used_code, use_raw,
            ),
        )
        logger.info(st_dataset)
        self.train_batch(
            epochs=epochs, loaders=loaders, call_func=self._loss_mixer, writer=writer
        )  # type:ignore
        dataset.st_adata.obsm["deconv"] = self.get_prop(
            st_dataset.rep.to(self.device),
            colnames=celltypes,
            rownames=st_adata.obs_names,
            phat=True,
        )
        dataset.st_adata.obsm["deconv_abundance"] = self.get_prop(
            st_dataset.rep.to(self.device),
            phat=True,
            norm=False,
            colnames=celltypes,
            rownames=st_adata.obs_names,
        )
        dataset.switch_layer(dataset.layer_key)
        self._get_px_r = _get_px_r
        self.st_decoder = st_decoder
        self.st_decoder_type = st_decoder_type
        self.model.to('cpu')
        self.st_decoder.to('cpu')
        self.mixer.to('cpu')

    def _make_collate_fn(self, batch_used_code, use_raw):
        if use_raw is False:
            return _collate_fn(batch_used_code)
        return default_collate

    def _fix_params(self):
        self.mixer.train()
        self.mixer.to(self.device)
        self.model._smooth = False
        self.model.eval()
        self.model.to(self.device)
        self.st_decoder.to(self.device)
        self.st_decoder.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.st_decoder.parameters():
            param.requires_grad = False

    def _get_ct_sig(self, adata, dataset, cell_type_key, use_raw, batch_used):
        if use_raw is not True:
            logger.info("Using model-based signatures")
            signatures = self.get_signatures(
                dataset=dataset,
                batch_used=batch_used,
            )
            if isinstance(batch_used, str) and batch_used not in ["all", "self"]:
                batch_used_code = dataset.batch_codes[batch_used]
            else:
                batch_used_code = batch_used
        else:
            logger.info("Using raw data-based signatures")
            signatures = torch.from_numpy(adata.layers[dataset.layer_key])
            signatures = signatures / signatures.sum(-1, keepdim=True)
            batch_used_code = None

        signatures, celltypes = aver_items_by_ct(
            adata, cell_type_key, signatures, weight=None, return_cts=True
        )

        return signatures, batch_used_code, celltypes

    def _assert_rep(self, adata, dataset, rep_key):
        if rep_key is not None:
            logger.info(f"Using precomputed embedding: {rep_key}")
            if rep_key not in adata.obsm_keys():
                logger.info(f"{rep_key} not found, back to gernerated")
                rep = dataset.rep
            else:
                rep = torch.from_numpy(adata.obsm[rep_key])
        else:
            rep = dataset.rep
            if rep is None:
                rep = self.embed(dataset, tsfmr_out=True, as_numpy=False)
                dataset.set_rep(rep)
        assert rep is not None, "No representation found"
        return rep

    def _set_st_counts(self, adata, dataset, library_size, use_raw, batch_used):
        _get_px_r = self._get_px_r
        st_decoder = self.st_decoder
        st_decoder_type = self.st_decoder_type
        if use_raw is False:
            logger.info("Using corrected counts for ST data")
            self.regress_out(adata, dataset, library_size=library_size, batch_used=batch_used)
            self.st_decoder = self.model.decoder
            self.st_decoder_type = self.model.decoder_type
            dataset.switch_layer("corrected_counts")
        else:
            logger.info("Using raw counts for ST data")
        return _get_px_r, st_decoder, st_decoder_type

    def nnls_deconv(
        self,
        adata: AnnData,
        dataset: CrossDataset,
        cell_type_key: Optional[str] = None,
        batch_used: Optional[str | int] = 0,
        n_jobs: int = 1,
    ):
        """Perform cell-type deconvolution on spatial transcriptomics data using non-negative least squares (NNLS).

        Args:
            adata (AnnData): Annotated data matrix.
            dataset (CrossDataset): CrossDataset object containing spatial transcriptomics data.
            cell_type_key (str, optional): Key for cell type annotation in the dataset. Default is None.
            batch_used (Optional[str | int]): The batch used for deconvolution. Default is 0.
            n_jobs (int): Number of jobs to run in parallel. Default is 1.
        """

        if cell_type_key is None:
            cell_type_key = dataset.class_key

        st_adata = dataset.st_adata

        self.regress_out(adata, dataset, batch_used=batch_used)
        celltypes = dataset.adata.obs[cell_type_key].cat.categories
        signatures = adata.layers["corrected_counts"]
        signatures, celltypes = aver_items_by_ct(
            adata, cell_type_key, signatures, return_cts=True
        )
        observations = adata[adata.obs['modality'] == 'ST'].layers["corrected_counts"]
        abundance = nnls_deconv(
            signatures=signatures,
            observations=observations,
            n_jobs=n_jobs,
        )

        st_adata.obsm["deconv_nnls"] = pd.DataFrame(
            data=abundance, columns=celltypes, index=st_adata.obs_names
        )

    def reference_map(
        self,
        adata,
        dataset,
        k=6,
        use_rep="X_rep",
        library_id=None,
    ):
        """
        Maps the reference dataset to the target dataset using k-nearest neighbors (KNN) algorithm.

        Args:
            adata (AnnData): The target dataset.
            dataset (Dataset): The reference dataset.
            k (int, optional): The number of nearest neighbors to consider. Defaults to 6.
            use_rep (str, optional): The representation to use for KNN. Defaults to 'X_rep'.
            library_id (str, optional): The library ID of the reference dataset. Defaults to None.

        Returns:
            AnnData: The target dataset with KNN information added.
        """
        if library_id is None:
            logger.info("Specify a library_id")
        if use_rep not in adata.obsm_keys():
            logger.info("No representation provided, generating embedding 'X_rep'")
            self.add_embed(adata, dataset)
            use_rep = "X_rep"
        st_adata = dataset.st_adata
        batch_key = dataset.batch_key
        adata = adata[
            (adata.obs["modality"] == "SC") | (
                adata.obs[batch_key] == library_id)
        ]
        Y_spatial = st_adata[st_adata.obs[batch_key]
                             == library_id].obsm["spatial"]

        knn_adata = knn_random_coords(
            adata=adata,
            k=k,
            Y_spatial=Y_spatial,
            use_rep=use_rep,
            library_id=library_id,
        )
        return knn_adata

    def run(self, **kwargs):
        raise NotImplementedError

    def get_pz(self, rep, colnames):
        df = self.get_prop(self, rep, phat=False)
        return df

    def get_prop(self, rep, colnames, rownames, phat=True, norm=True, ct_scale=True):
        """Get the proportions of the given representation.

        Args:
            rep (torch.Tensor): The input representation.
            colnames (list): The column names for the resulting DataFrame.
            rownames (list): The row names for the resulting DataFrame.
            phat (bool, optional): Whether to use the estimated proportions. Defaults to True.
            norm (bool, optional): Whether to normalize the proportions. Defaults to True.
            ct_scale (bool, optional): Whether to scale the proportions by the cell type scaling factors. Defaults to True.

        Returns:
            pd.DataFrame: The DataFrame containing the proportions.
        """
        self.mixer.eval()
        self.mixer.to(self.device)
        with torch.no_grad():
            prop = self.mixer.get_prop(
                rep, phat=phat, ct_scale=ct_scale).cpu().numpy()
            if norm:
                prop /= prop.sum(-1, keepdims=True) + 1e-8
            data = prop
        df = pd.DataFrame(data=data, columns=colnames, index=rownames)
        self.mixer.cpu()
        return df

    @torch.no_grad()
    def get_scores(self, rep, colnames, rownames):
        """Get the scores for the given representation.

        Args:
            rep (torch.Tensor): The representation tensor.
            colnames (list): The column names for the resulting DataFrame.
            rownames (list): The row names for the resulting DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the scores with the specified column and row names.
        """
        scores = self.mixer.get_scores(
            rep.to(self.device),
        )
        data = scores.cpu().numpy()
        return pd.DataFrame(data=data, columns=colnames, index=rownames)

    def st_impute(
        self,
        dataset: CrossDataset,
        key_added="st_expected_counts",
        smooth=False,
        qc=False,
    ):
        """Perform imputation on spatial transcriptomics data.

        Args:
            dataset (CrossDataset): The dataset containing spatial transcriptomics data.
            key_added (str, optional): The key to store the imputed counts in the dataset. Defaults to 'st_expected_counts'.
            smooth (bool, optional): Whether to apply smoothing during imputation. Defaults to False.
            qc (bool, optional): Whether to perform quality control during imputation. Defaults to False.
        """

        assert dataset.rep is not None, "add rep to dataset first"
        st_dataset = dataset.subset_st()
        self.model._smooth = smooth
        self.model = self.model.to(self.device)
        self.st_decoder = self.st_decoder.to(self.device)
        cls_rep = self.model.readout_(st_dataset.rep.to(self.device))
        decode_dict = self._st_decode(
            cls_rep,
            st_dataset.gene_expr.to(self.device),
            self.get_batch_ohenc(
                st_dataset.batch_label
            ).to(self.device),
        )
        self.impute(
            adata=dataset.st_adata,
            dataset=st_dataset,
            decode_dict=decode_dict,
            key_added=key_added,
            qc=qc,
        )
        self.model.cpu()
        self.st_decoder.cpu()

    @torch.no_grad()
    def get_signatures(
        self,
        dataset: CrossDataset,
        batch_used: str | int | None = 0,
    ) -> torch.Tensor:
        """Get the signatures for the given dataset.

        Args:
            dataset (CrossDataset): The dataset containing the signatures.
            batch_used (str, int, optional): The batch used for obtaining the signatures. Defaults to 0.

        Returns:
            torch.Tensor: The signatures for the given dataset.
        """
        if batch_used == 'self':
            signatures = self.impute(
                adata=dataset.adata,
                dataset=dataset,
                key_added=None,
                qc=False,
            )
            signatures = torch.from_numpy(signatures)
            return signatures / signatures.sum(-1, keepdim=True)
        return super().get_signatures(dataset, batch_used)

    def load_checkpoint(self, checkpoint: str | Path | dict):
        if not isinstance(checkpoint, dict):
            super().load_checkpoint(checkpoint)
        else:
            super().load_checkpoint(checkpoint["model"])

            st_decoder_config = checkpoint.get("st_decoder_config", None)
            if st_decoder_config is not None:
                self._init_st_decoder(**st_decoder_config)
                if st_decoder_config.get("use_st_decoder", False):
                    logger.info("Loading st_decoder...")
                    self.st_decoder.load_state_dict(torch.load(checkpoint["st_decoder"]))
                    logger.info("Load st_decoder done")

            # mixer_config = checkpoint.get("mixer_config", None)
            # if mixer_config is not None:
            #     logger.info("Loading mixer...")
            #     self._init_mixer(**mixer_config)
            #     self.mixer.load_state_dict(torch.load(checkpoint["mixer"]))
            #     logger.info("Load mixer done")


def _contrastive_loss(rep_q, rep_k):
    logits = torch.matmul(F.normalize(rep_q), F.normalize(rep_k).T)
    device = rep_q.device
    labels = torch.zeros(rep_k.shape[0]).long().to(device)
    contrast_loss = 0.7 * F.cross_entropy(logits / 0.1, labels)
    return contrast_loss


def _collate_fn(label):
    def _collate(batch):
        batch = default_collate(batch)
        transfered = torch.ones_like(batch[-1]) * label
        return (*batch[:-1], transfered)
    return _collate
