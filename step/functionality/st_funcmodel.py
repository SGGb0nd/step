from functools import partial
from typing import Optional

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from torch.distributions import kl_divergence as kl

from step.manager import logger
from step.models.geneformer import Geneformer
from step.utils.dataset import StDataset
from step.utils.gbolt import MultiGraphsAllNodesSampler
from step.utils.misc import generate_adj

from ..models.extension import NrmlsBC
from .base import FunctionalBase


class stSmoother(FunctionalBase):
    """
    A class for training the model for identifying spatial domains.

    Attributes:

        model (NrmlsBC): The model to be trained.
        use_earlystop (bool): Whether to use early stopping.
        _factor (float): The beta value for training.
        _num_batches (int): The number of batches.
        _from_scmodel (bool): Whether the model is from a scModel object.
        _gener_graph (callable): A function for generating the graph.
    """

    def __init__(self,
                 num_batches=1,
                 beta=1e-2,
                 variational=False,
                 dispersion="batch-gene",
                 n_glayers=4,
                 device=None,
                 **kwargs):
        """
        Initialize the stSmoother object.

        Args:
            num_batches (int, optional): The number of batches. Defaults to 1.
            beta (float, optional): The beta value for training. Defaults to 1e-2.
            variational (bool, optional): Whether to use variational training. Defaults to False.
            **kwargs: Additional keyword arguments.

        """
        use_earlystop = kwargs.pop("use_earlystop", False)
        self.max_neighs = kwargs.pop("max_neighbors", 10)
        self.edge_clip = kwargs.get("edge_clip", None)
        use_l_scale = kwargs.get("use_l_scale", True)
        if num_batches > 1:
            model = NrmlsBC(
                num_batches=num_batches,
                num_classes=0,
                variational=variational,
                dispersion=dispersion,
                n_glayers=n_glayers,
                use_l_scale=use_l_scale,
                **kwargs
            )
        else:
            model = Geneformer(variational=variational,
                               n_glayers=n_glayers,
                               **kwargs)

        model.init_smoother_with_builtin()
        super().__init__(model=model, use_earlystop=use_earlystop, device=device)
        self._factor = beta
        self._num_batches = num_batches
        self._gener_graph = partial(
            generate_adj, edge_clip=self.edge_clip, max_neighbors=self.max_neighs
        )
        self.loss = self._loss_no_g

    def from_spatial_graph(
        self,
        adata: AnnData,
        dataset: StDataset,
        batch_size: int = 1024,
        num_iters=1000,
        batch_key: Optional[str] = None,
        sample_rate: float = 1.,
    ):
        """
        Convert spatial graph data into a dataloader for training.

        Args:
            adata (AnnData): Annotated data object containing spatial graph information.
            dataset (StDataset): Spatial transcriptomics dataset object.
            batch_size (int, optional): Number of samples per batch. Defaults to 1024.
            num_iters (int, optional): Number of iterations. Defaults to 1000.
            batch_key (str, optional): Key for batch information in `adata.obs`. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dataloader: A dataloader object for training.

        """
        if self._num_batches > 1:
            graphs = []
            indcies = np.arange(len(adata))
            if batch_key is None:
                batch_key = dataset.batch_key
            for batch in adata.obs[batch_key].cat.categories:
                index = indcies[adata.obs[batch_key] == batch]
                _adata = adata[index]
                g = self._gener_graph(_adata)
                g.ndata["feat"] = dataset.get(batch, "gene_expr")
                g.ndata["batch_label"] = dataset.get(batch, "batch_label")
                graphs.append(g)
            batched_graph = dgl.batch(graphs)
        else:
            batched_graph = self._gener_graph(adata)
            batched_graph.ndata["feat"] = dataset.gene_expr

        if self._num_batches > 1:
            if not self.e2e:
                dataloader = dgl.dataloading.GraphDataLoader(
                    dgl.unbatch(batched_graph),
                    batch_size=batch_size,
                    num_workers=1,
                )
                return dataloader

            sampler = MultiGraphsAllNodesSampler(
                mode="node",
                budget=batch_size,
                n_graphs=batch_size,
                ratio=sample_rate,
            )
            dataloader = dgl.dataloading.DataLoader(
                batched_graph,
                torch.arange(num_iters),
                sampler,
                num_workers=1,
            )
        else:
            sampler = dgl.dataloading.SAINTSampler(
                mode="node", budget=batch_size)
            dataloader = dgl.dataloading.DataLoader(
                batched_graph,
                torch.arange(num_iters),
                sampler,
                num_workers=1,
            )
        return dataloader

    def _loss_g(self, x, step, batch_rep=None, rep=None, ind=None):
        """
        Calculate the loss for the generator network.

        Args:
            x: Input data.
            step: Training step.
            batch_rep: Batch representation.
            rep: Representation.
            ind: Index.

        Returns:
            dict: The loss dictionary.
                recon_loss (torch.Tensor): The reconstruction loss.
                kl_loss (torch.Tensor): The KL loss.
                contrast_loss (optional, torch.Tensor): The contrastive loss.
        """
        rep_ts = (
            self.model.encode_ts(
                x, batch_rep)
        )
        rep_q = self.model.local_smooth(rep_ts)
        cls_rep = self.model.readout(rep_q)
        if self.contrast and self._kl_contrast:
            dist = self.model.readout.dist

        decode_dict = self.model.decode(cls_rep, x, batch_rep=batch_rep)
        loss_dict = self.loss_fn(**decode_dict)

        if self.e2e:
            rep_k = rep_ts
            decode_dict2 = self.model.decode_ts(rep_ts, x, batch_rep=batch_rep)
            if self.contrast:
                if self._kl_contrast:
                    dist2 = self.model.readout.dist
                    contrast_loss = kl(dist, dist2).mean() * 0.01
                else:
                    contrast_loss = _contrastive_loss(
                        rep_q, rep_k, self.device)
                loss_dict["contrast_loss"] = contrast_loss

            recon_loss2 = self.loss_fn(nokl=True, **decode_dict2)["recon_loss"]
            loss_dict["recon_loss"] = loss_dict["recon_loss"] + recon_loss2
        return loss_dict

    def _loss_no_g(self, x, batch_rep=None, rep=None, ind=None):
        """
        Calculate the loss without the gradient term.

        Args:
            x: Input data.
            batch_rep: Batch representation.
            rep: Representation.
            ind: Index.

        Returns:
            The loss value.
        """
        rep_ts = self.model.encode_ts(x, batch_rep)
        decode_dict = self.model.decode_ts(rep_ts, x_gd=x, batch_rep=batch_rep)
        return self.loss_fn(**decode_dict)

    def loss_gbatch(self, g, x_gd, batch_rep, step, ind=None):
        """
        Compute the loss for a batch of graphs.

        Args:
            g (dgl.DGLGraph): The input graph.
            x_gd (torch.Tensor): The input graph features.
            batch_rep (torch.Tensor): The batch representation.
            step (int): The current training step.
            ind (torch.Tensor, optional): The indices of the nodes to compute the loss for.

        Returns:
            dict: The loss dictionary.
        """
        self.model: NrmlsBC
        self.model._smooth = True
        g = dgl.add_self_loop(g)
        self.model.g = g
        rep = g.ndata.get("rep", None)
        return self._loss_g(x_gd, step=step, batch_rep=batch_rep, rep=rep, ind=ind)

    def handle_input_tuple(self, input_tuple, ind=None, rep=None,):
        """
        Handles the input tuple and computes the loss.

        Args:
            input_tuple (tuple): The input tuple containing the data and batch label.
            ind (int, optional): The index. Defaults to None.
            rep (int, optional): The representation. Defaults to None.

        Returns:
            dict: The loss dictionary.
        """
        if self._num_batches > 1:
            X, batch_label = input_tuple
        else:
            X = input_tuple
            batch_label = None
        batch_rep = self.get_batch_ohenc(batch_label, device=self.device)
        loss_dict = self._loss_no_g(
            X.to(self.device),
            ind=ind,
            batch_rep=batch_rep,
            rep=rep,
        )
        return loss_dict

    def handle_ginput_tuple(self, input_tuple, ind, step=None):
        """
        Handles the input tuple for graph data and performs necessary operations.

        Args:
            input_tuple (tuple): The input tuple containing the graph data.
            ind (int): The index.
            step (int, optional): The step. Defaults to None.

        Returns:
            dict: The loss dictionary.
        """

        g = input_tuple
        g = g.to(self.device)
        x_gd = g.ndata["feat"]
        ind = g.ndata.get("ind", None)
        pos_enc = g.ndata.get("pos_enc", None)
        if pos_enc is not None:
            g.ndata["pos_enc"] = pos_enc.to(self.device)
        batch_label = g.ndata.get("batch_label", None)
        batch_rep = None
        if self._num_batches > 1:
            batch_rep = F.one_hot(
                batch_label, num_classes=self._num_batches).float()
            batch_rep = batch_rep.to(self.device)
        loss_dict = self.loss_gbatch(
            g=g,
            x_gd=x_gd.to(self.device),
            batch_rep=batch_rep,
            ind=ind,
            step=step,
        )
        loss_dict["graph_ids"] = torch.unique(
            batch_label
        ).cpu().numpy().tolist() if batch_label is not None else None
        self.model.g = self.model.g.to("cpu")
        return loss_dict

    @torch.no_grad()
    def _embed(self, dataset: StDataset, tsfmr_out=False, smooth=False, as_numpy=True):
        """
        Embeds the given dataset using the model.

        Args:
            dataset (StDataset): The dataset to be embedded.
            tsfmr_out (bool, optional): Whether to use the transformer output for embedding. Defaults to False.
            smooth (bool, optional): Whether to apply smoothing during embedding. Defaults to False.
            as_numpy (bool, optional): Whether to return the embedding as a numpy array. Defaults to True.

        Returns:
            torch.Tensor or numpy.ndarray: The embedded representation of the dataset.
        """

        self.model.eval()
        if getattr(self.model, "device", None) != self.device:
            self.model.to(self.device)
        self.model._smooth = smooth
        adata = dataset.adata
        if self._num_batches > 1:
            dim = self.model.module_dim if not tsfmr_out else self.model.hidden_dim
            rep = torch.zeros(len(dataset), dim)
            fn = self.model.encode if not tsfmr_out else self.model.encode_ts
            batch_key = dataset.batch_key
            for batch in adata.obs[batch_key].cat.categories:
                _adata = adata[adata.obs[batch_key] == batch]
                if smooth:
                    g = self._gener_graph(_adata)
                    self.model.g = g.to(self.device)
                x = dataset.get(batch, "gene_expr")
                batch_rep = dataset.get(batch, "batch_label")
                batch_rep = F.one_hot(
                    batch_rep, num_classes=self._num_batches).float()
                index = dataset.get(batch, attr=None)
                rep[index] = fn(
                    x.to(self.device), batch_rep=batch_rep.to(self.device)
                ).cpu()
            if smooth:
                self.model.g = self.model.g.to("cpu")
            if as_numpy:
                return rep.numpy()
            return rep

        if smooth:
            g = self._gener_graph(adata)
            self.model.g = g.to(self.device)
        rep = super().embed(dataset, tsfmr_out=tsfmr_out, as_numpy=as_numpy)
        if smooth:
            self.model.g = self.model.g.to("cpu")
        return rep

    @torch.no_grad()
    def embed(self, dataset: StDataset, tsfmr_out=False, as_numpy=True):
        """
        Embeds the given dataset using the model.

        Args:
            dataset (StDataset): The dataset to be embedded.
            tsfmr_out (bool, optional): Whether to return the output of the transformer layer. Defaults to False.
            as_numpy (bool, optional): Whether to return the embeddings as numpy arrays. Defaults to True.

        Returns:
            The embeddings of the dataset.
        """

        return self._embed(
            dataset, tsfmr_out=tsfmr_out, as_numpy=as_numpy, smooth=False
        )

    @torch.no_grad()
    def gembed(self, dataset: StDataset, tsfmr_out=False, as_numpy=True):
        """
        Embeds the given dataset using the current model.

        Args:
            dataset (StDataset): The dataset to be embedded.
            tsfmr_out (bool, optional): Whether to return the output of the transformer layer. Defaults to False.
            as_numpy (bool, optional): Whether to return the embeddings as numpy arrays. Defaults to True.

        Returns:
            The embeddings of the dataset.
        """

        return self._embed(dataset, tsfmr_out=tsfmr_out, as_numpy=as_numpy, smooth=True)

    def run(
        self,
        adata,
        dataset: StDataset,
        epochs=1,
        batch_size=1024,
        graph_batch_size=1,
        smooth_epochs=1,
        n_samples=1024,
        n_iterations=2000,
        split_rate=0.0,
        sample_rate=1.,
        beta=1e-3,
        key_added="X_smoothed",
        lr=1e-3,
        tune_lr=1e-5,
        kl_cutoff=None,
        e2e=True,
        contrast=True,
        kl_contrast=False,
        logging=False,
        reset=False,
    ):
        """
        Runs the training process for the model.

        Args:
            adata: The AnnData object containing the data.
            dataset: The StDataset object containing the spatial data.
            epochs: The number of training epochs (default: 1).
            batch_size: The batch size for training (default: 1024).
            graph_batch_size: The batch size for graph training (default: 1).
            n_samples: The number of samples for training (default: 1024).
            n_iterations: The number of iterations for training (default: 2000).
            split_rate: The split rate for training (default: 0.).
            beta: The beta value for training (default: 1e-3).
            key_added: The key to add the smoothed data to the AnnData object (default: 'X_smoothed').
            lr: The learning rate for training (default: 1e-3).
            tune_lr: The learning rate for fine-tuning (default: 1e-5).
            kl_cutoff: The KL cutoff value for training (default: None).
            e2e: Whether to use end-to-end training (default: True).
            contrast: Whether to use contrastive loss (default: True).
            kl_contrast: Whether to use KL contrastive loss (default: False).
            logging: Whether to enable logging (default: False).
            kl_anneal: Whether to anneal the KL loss (default: False).

        """
        if reset:
            self.reset_model()
        writer_gcn = None
        if logging:
            from torch.utils.tensorboard import SummaryWriter
            writer_gcn = SummaryWriter(comment="-st-domains-gcn")

        setattr(self, "e2e", e2e)
        setattr(self, "_kl_contrast", kl_contrast)
        self.set_kl_cutoff(kl_cutoff)
        setattr(self, "contrast", contrast)
        if getattr(self.model, "device", None) != self.device:
            self.model.to(self.device)
        if not e2e:
            logger.info("Training with 2 stages pattern: 1/2")
            self.model._smooth = False
            batch_size = len(dataset) if batch_size is None else batch_size
            loaders = self.make_loaders(dataset, batch_size, split_rate)
            self.train_batch(epochs=epochs, loaders=loaders)  # type:ignore

            self.init_optimizer(
                stage=2,
                lr=lr,
                tune_batch_embedding=False,
                tune_lr=tune_lr,
            )
        else:
            logger.info("Training with e2e pattern")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model._smooth = True
        if beta is not None:
            self.set_beta(beta)
        if self._num_batches > 1:
            logger.info("Training graph with multiple batches")
            gloader = self.from_spatial_graph(
                adata=adata,
                dataset=dataset,
                batch_size=graph_batch_size,
                num_iters=n_iterations,
                sample_rate=sample_rate,
            )
            if self.e2e:
                self.train_node_sampler(
                    epochs=1,  # type:ignore
                    gloader=gloader,
                    writer=writer_gcn,
                )
                rep = self.gembed(dataset)
            else:
                logger.info("Training with 2 stages pattern: 2/2")
                self.train_graph_batch(
                    epochs=smooth_epochs,  # type:ignore
                    gloader=gloader,
                    writer=writer_gcn,
                )
                rep = self.gembed(dataset)
        else:
            logger.info("Training graph with single batch")
            if n_samples is not None:
                gloader = self.from_spatial_graph(
                    adata=adata,
                    dataset=dataset,
                    batch_size=n_samples,
                    num_iters=n_iterations,
                    sample_rate=sample_rate,
                )
                self.train_node_sampler(
                    epochs=1,  # type:ignore
                    gloader=gloader,
                    writer=writer_gcn,
                )
            else:
                logger.info("Training with 2 stages pattern: 2/2")
                self.train(
                    epochs=smooth_epochs,  # type:ignore
                    X=dataset.gene_expr,
                    call_func=self._loss_g,
                )
            rep = self.gembed(dataset)
        if key_added is not None:
            adata.obsm[key_added] = rep
            if not e2e:
                unsmoothed_rep = self.embed(dataset)
                adata.obsm["X_unsmoothed"] = unsmoothed_rep
        self.model.cpu()

    def reset_model(self):
        super().reset_model()
        self.model.init_smoother_with_builtin()


def _contrastive_loss(rep_q, rep_k, device):
    logits = torch.matmul(F.normalize(rep_q), F.normalize(rep_k).T)
    labels = torch.zeros(rep_k.shape[0]).long().to(device)
    contrast_loss = 0.7 * F.cross_entropy(logits / 0.1, labels)
    return contrast_loss
