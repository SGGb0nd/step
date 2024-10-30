from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from torch.distributions import Poisson

from step.manager import logger
from step.models.distributions import (NegativeBinomial,
                                       ZeroInflatedNegativeBinomial)
from step.models.transcriptformer import TranscriptFormer, Readout
from step.utils.dataset import BaseDataset


class ModelOps(object):
    """A class for wrapping a TranscriptFormer model with useful methods.

    Attributes:
        model (TranscriptFormer): The TranscriptFormer model.

    """

    def __init__(
        self,
        model: TranscriptFormer,
        device: str | None,
    ):
        """
        Take a TranscriptFormer model and wrap it with some useful methods.

        Args:
            model (TranscriptFormer): a TranscriptFormer model
            _factor (float): a factor to scale the loss
            kl_cutoff (float): a cutoff value for the KL loss
        """
        self.model = model

        # self._factor = 0.01 if model.decoder_type == 'zinb' else 1e-3
        self._factor = 1e-2
        self.kl_cutoff = None
        self.kl_start_pg = 0.9
        self._kl_cutfn: Callable
        self.device = device

    def set_kl_cutoff(self, kl_cutoff):
        """Sets the KL cutoff value for the model.

        Args:
            kl_cutoff (float): The cutoff value for the KL loss.

        Returns:
            None
        """
        self.kl_cutoff = kl_cutoff if kl_cutoff is not None else 0
        cutoff = self.kl_cutoff
        if cutoff is not None and cutoff > 0:
            self._kl_cutfn = lambda kl_loss: torch.where(
                kl_loss < cutoff, torch.zeros_like(
                    kl_loss).to(self.device), kl_loss
            )
            # self._kl_cutfn = lambda kl_loss: 0 if kl_loss <= cutoff else kl_loss
        else:
            self._kl_cutfn = lambda x: x

    def _get_kl_weight(self, step=None):
        """Calculate the KL weight based on the current step.

        Args:
            step (int): The current step.

        Returns:
            float: The KL weight.

        """
        kl_start = getattr(self, "kl_start", None)
        if (kl_start is None) or (step is None):
            return self._factor

        kl_weight = torch.sigmoid(torch.tensor(
            self._factor * (step - kl_start)))
        return float(kl_weight)

    def loss(self, x, ind=None):
        """loss function for a single batch

        Args:
            x (torch.Tensor): gene expression matrix

        Returns:
            loss_dict (dict): a dictionary of loss values
        """
        rep_ts = self.model.encode_ts(x)
        if ind is not None:
            rep_ts = rep_ts[ind]
            x = x[ind]
        decode_dict = self.model.decode_ts(
            rep_ts=rep_ts,
            x_gd=x,
        )
        loss_dict = self.loss_fn(**decode_dict)
        del decode_dict
        return loss_dict

    def loss_gbatch(self):
        """Loss function for a graph batch

        Should be implemented in the derived class.

        Returns:
            loss_dict (dict): a dictionary of loss values
        """
        pass

    def get_px(self, decode_dict):
        """Get the probability distribution for the observed data.

        Args:
            decode_dict (dict): A dictionary containing the decoding parameters.

        Returns:
            px: The probability distribution for the observed data.
        """

        if decode_dict["decoder_type"] == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=decode_dict["px_rate"],
                theta=decode_dict["px_r"].exp(),
                zi_logits=decode_dict["px_dropout"],
                scale=decode_dict.get("px_scale", None),
            )
        else:
            px = NegativeBinomial(
                mu=decode_dict["px_rate"],
                theta=decode_dict["px_r"],
                scale=decode_dict.get("px_scale", None),
            )
        return px

    def get_batch_ohenc(self, batch_label: Optional[torch.Tensor], average=False, device=None):
        """Converts the batch labels/indicators into one-hot encoded format.

        Args:
            batch_label (Tensor): The batch labels.

        Returns:
            batch_oh (Tensor): The one-hot encoded batch labels.
        """
        num_batches = getattr(self, "_num_batches", 1)
        if num_batches == 1 or batch_label is None:
            return None

        if average:
            batch_oh = torch.ones(
                batch_label.shape[0], num_batches) / num_batches
            if device is not None:
                batch_oh = batch_oh.to(device)
            return batch_oh
        batch_oh = F.one_hot(batch_label.long(),
                             num_classes=num_batches).float()
        if device is not None:
            batch_oh = batch_oh.to(device)
        return batch_oh

    @torch.no_grad()
    def embed(
        self, dataset: BaseDataset, tsfmr_out=False, as_numpy=True
    ):
        """get the representation of the dataset

        Args:
            dataset (ScDataset): ScDataset object for scRNA-seq data
            as_numpy (bool, optional): whether to return the representation as a numpy array. Defaults to True.

        Returns:
            rep (torch.Tensor): the representation of the dataset
        """
        self.model.eval()
        self.model.to(self.device)
        x = dataset.gene_expr
        if not tsfmr_out:
            rep = self.model.encode(x.to(self.device)).cpu()
        else:
            rep = self.model.encode_ts(x.to(self.device)).cpu()
        self.model.cpu()
        if as_numpy:
            return rep.numpy()
        return rep

    def add_embed(self, adata: AnnData, dataset: BaseDataset, tsfmr_out=False, key_added="X_rep"):
        """add the representation of the dataset to the adata object

        Args:
            adata (AnnData): an AnnData object
            dataset (BaseDataset): _description_
            key_added (str, optional): _description_. Defaults to 'X_rep'.
        """
        rep = self.embed(dataset=dataset, tsfmr_out=tsfmr_out)
        if tsfmr_out:
            key_added = "X_rep_tsfmr" if key_added is None else key_added
        adata.obsm[key_added] = rep

    @torch.no_grad()
    def get_signatures(
        self,
        dataset,
        batch_used,
    ) -> torch.Tensor:
        """Retrieves the signatures from the model for the given dataset.

        Args:
            dataset: The dataset for which to retrieve the signatures.
            batch_label: The batch label to use for signature retrieval. If None, the batch label from the dataset will be used.
            use_batch_rep: Whether to use batch representation for signature retrieval.

        Returns:
            The retrieved signatures.

        """
        return self.regress_out(
            adata=dataset.adata,
            dataset=dataset,
            batch_used=batch_used,
            library_size=1,
            key_added=None,
        )  # type: ignore

    def set_kl_start(self, n_iterations: int):
        setattr(self, "kl_start", int(self.kl_start_pg * n_iterations))

    @torch.no_grad()
    def impute(
        self,
        adata: Optional[AnnData] = None,
        dataset: Optional[BaseDataset] = None,
        batch_label: Optional[int] = None,
        decode_dict=None,
        key_added: str | None = "expected_counts",
        rep_key: str | None = "X_rep",
        layer_key: Optional[str] = None,
        rep: torch.Tensor | None = None,
        x=None,
        qc=False,
        return_counts=False,
        log10=True,
    ) -> np.ndarray | None:
        """Imputes missing values in the gene expression data.

        This process involves using the trained model to predict the gene expression values for the missing data by using the mean of the distribution of the gene expression values.

        Args:
            adata (AnnData): Annotated data object containing the gene expression data.
            dataset (ScDataset): Single-cell dataset object containing the gene expression data.
            batch_label (int): The batch label to use for signature retrieval. If None, the batch label from the dataset will be used.
            decode_dict (dict): Dictionary containing the decoded gene expression values.
            key_added (str): Key to add the imputed gene expression values to in the adata object.
            rep_key (str): Key to access the representation data in the adata object.
            layer_key (str): Key to access the layer data in the dataset object.
            rep (ndarray): Representation data.
            x (ndarray): Gene expression data, only used for library size or sequencing depth normalization.
            qc (bool): Flag indicating whether to perform quality control.
            return_counts (bool): Flag indicating whether to return the imputed gene expression counts.
            log10 (bool): Flag indicating whether to perform log10 transformation in the quality control plot.

        Returns:
            ndarray: Imputed gene expression data.
        """

        self.model.eval()
        if rep_key and adata is not None:
            rep = torch.from_numpy(adata.obsm[rep_key])
        if dataset is None:
            assert (rep is not None and x is not None) or decode_dict is not None
        else:
            x = dataset.gene_expr

        if decode_dict is None:
            batch_labels = torch.ones(len(rep), dtype=torch.long) * batch_label  # type: ignore
            batch_rep = self.get_batch_ohenc(batch_labels)
            decode_dict = self.model.decode(
                cls_rep=rep.to(self.device),
                x_gd=x.to(self.device),
                batch_rep=batch_rep.to(self.device),
            )
        px = self.get_px(decode_dict)
        x_recon = px.mean

        if adata is None or return_counts or key_added is None:
            return x_recon.cpu().numpy()
        adata.layers[key_added] = x_recon.cpu().numpy()
        if qc:
            if layer_key is None:
                layer_key = dataset.layer_key
            from step.utils.plotting import plot_posterior_mu_vs_data

            plot_posterior_mu_vs_data(
                adata.layers[key_added],
                adata.layers[layer_key],
                log10=log10,
            )

    @torch.no_grad()
    def generate(
        self,
        adata: Optional[AnnData] = None,
        dataset: Optional[BaseDataset] = None,
        key_added="generated_counts",
        rep_key="X_rep",
        rep=None,
        x=None,
    ):
        """Generate synthetic data using the trained model.

        This process involves using the trained model(distribution) to generate synthetic gene expression data.

        Args:
            adata (AnnData): Annotated data object containing the representation matrix.
            dataset (ScDataset): Single-cell dataset object containing the gene expression data.
            key_added (str): Key to store the generated counts in the `adata.layers` attribute.
            rep_key (str): Key to access the representation matrix in `adata.obsm`.
            rep (ndarray): Representation matrix. If not provided, it will be retrieved from `adata.obsm`.
            x (ndarray): Gene expression data. If not provided, it will be retrieved from `dataset.gene_expr`.

        Returns:
            ndarray: Generated synthetic data as a numpy array.

        """

        self.model.eval()
        if rep_key and adata is not None:
            rep = adata.obsm[rep_key]
            rep = torch.from_numpy(rep)
        if dataset is None:
            assert rep is not None and x is not None
        else:
            x = dataset.gene_expr

        decode_dict = self.model.decode(
            cls_rep=rep.to(self.device),
            x_gd=x.to(self.device),
        )
        px = self.get_px(decode_dict)
        x_recon = px.sample()

        if adata is None:
            return x_recon.cpu().numpy()
        adata.layers[key_added] = x_recon.cpu().numpy()

    @torch.no_grad()
    def regress_out(
        self,
        adata: AnnData,
        dataset: BaseDataset,
        batch_used: Optional[Union[str, int]] = None,
        library_size: float = 1e5,
        key_added: str | None = "corrected_counts",
        rep_key="X_rep",
        rep=None,
    ) -> torch.Tensor | None:
        """Regress out the unwanted sources of variation from the gene expression data.

        This process involves using the trained model to predict the gene expression values for the unwanted sources of variation and then subtracting these values from the original gene expression data.

        Args:
            adata (AnnData): Annotated data object containing the gene expression data.
            dataset (ScDataset): Single-cell dataset object containing the gene expression data.
            batch_used (int): The batch label to use for signature retrieval. If None, the batch label from the dataset will be used.
            library_size (int): The library size or sequencing depth to use for normalization.
            key_added (str): Key to add the corrected gene expression values to in the adata object.
            rep_key (str): Key to access the representation data in the adata object.
            rep (ndarray): Representation data.

        Returns:
            ndarray: Regressed out gene expression data.

        """

        self.model.eval()
        if rep_key:
            rep = adata.obsm.get(rep_key, None)
            if rep is not None:
                assert rep.shape[1] == self.model.module_dim, f"Shape missmatched, expected dim of rep equal to `module_dim`: {self.model.module_dim}, got {rep.shape[1]}"
                rep = torch.from_numpy(rep)
            else:
                rep = self.embed(dataset, as_numpy=False)
        elif rep is not None:
            rep = torch.from_numpy(rep)
        else:
            rep = self.embed(dataset, as_numpy=False)

        x = dataset.gene_expr

        if batch_used != 'all':
            if isinstance(batch_used, str):
                assert batch_used in dataset.batch_codes, f"Invalid batch name: {batch_used}"
                batch_used = dataset.batch_codes[batch_used]
            batch_rep = self.get_batch_ohenc(
                torch.ones(len(dataset)) * batch_used
            )
        else:
            batch_rep = self.get_batch_ohenc(
                torch.ones(len(dataset)),
                average=True,
            )

        self.model.to(self.device)
        decode_dict = self.model.decode(
            cls_rep=rep.to(self.device),
            x_gd=x.to(self.device),
            batch_rep=batch_rep.to(self.device),
        )
        px = self.get_px(decode_dict)
        x_recon = px.mean
        if library_size is not None:
            logger.info("Library size normalization is performed.")
            x_recon = x_recon / x_recon.sum(1, keepdim=True) * library_size

        self.model.cpu()
        if key_added is None:
            return x_recon
        adata.layers[key_added] = x_recon.cpu().numpy()

    def show_attn_maps(self):
        raise NotImplementedError

    @torch.no_grad()
    def get_gene_modules(self, adata):
        """get the gene modules from the model

        Args:
            adata (_type_): _description_
        """
        self.model.eval()
        modules = self.model.moduler.extractor.weights.cpu().numpy()
        for i in range(modules.shape[-1]):
            adata.varm[f"modules_{i}"] = modules[..., i]

    def loss_fn(
        self,
        x: torch.Tensor,
        decoder_type: str,
        px_r: torch.Tensor,
        px_rate: torch.Tensor | None = None,
        px_dropout: torch.Tensor | None = None,
        px_scale: torch.Tensor | None = None,
        module: Readout | None = None,
        nokl=False,
        reduction="mean",
        **kwargs,
    ):
        """Loss function for a basic objective: reconstruction + KL divergence.

        This method computes the basic objective function for the model, which is the sum of the reconstruction loss and the KL divergence.
        The reconstruction loss is the negative log-likelihood of the observed data given the model's parameters, while the KL divergence is the Kullback-Leibler divergence between the approximate posterior and the prior.

        Args:
            px_rate (torch.Tensor): estimated mean of observations.
            px_dropout (torch.Tensor): logit of dropout rate.
            px_scale (torch.Tensor): normalized estimated mean of observations.
            px_r (torch.Tensor): over-dispersion parameter.
            x (torch.Tensor): gene expression matrix.
            decoder_type (str, optional): decoder type. Defaults to None.
            nokl (bool, optional): no kl loss. Defaults to False.
            reduction (str, optional): reduction method. Defaults to 'mean'.

        Returns:
            dict: A dictionary of loss values.
                recon_loss (torch.Tensor): Reconstruction loss.
                kl_loss (torch.Tensor): Scaled KL-divergence.
        """
        agg_func = torch.mean if reduction == "mean" else torch.sum
        step = kwargs.pop("step", None)

        if decoder_type == "zinb":
            loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate,
                    theta=px_r.exp(),
                    zi_logits=px_dropout,
                    scale=px_scale,
                    **kwargs,
                )
                .log_prob(x)
                .sum(-1)
            )
        elif decoder_type == "poisson":
            loss = -Poisson(rate=px_rate).log_prob(x).sum(-1)
        else:
            loss = (
                -NegativeBinomial(
                    mu=px_rate,
                    theta=px_r.exp(),
                )
                .log_prob(x)
                .sum(-1)
            )
        nokl = True if self._factor is None else nokl
        if module is None:
            module = self.model.readout

        if not module.variational or nokl:
            loss_dict = dict(
                recon_loss=agg_func(loss),
            )
        else:
            kl_loss = module.kl_loss()
            kl_loss = self._kl_cutfn(kl_loss).sum(-1)
            kl_weight = self._get_kl_weight(step=step)
            loss_dict = dict(
                recon_loss=agg_func(loss),
                kl_loss=kl_weight * agg_func(kl_loss),
            )
        return loss_dict

    def set_beta(self, beta: float):
        """Set the factor to scale the kl loss.

        Args:
            beta (float): The factor to scale the loss.
        """
        self._factor = beta

    def reset_model(
        self,
    ):
        """Reset the model parameters."""
        self.model.readout.clear()
        new_model = self.model.copy(with_state=False)
        self.model = new_model
        torch.cuda.empty_cache()
