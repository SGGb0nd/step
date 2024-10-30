from typing import Literal, Optional

import torch
import torch.nn.functional as F
from anndata import AnnData

from step.manager import logger
from step.models.transcriptformer import TranscriptFormer
from step.utils.dataset import BaseDataset, MaskedDataset, ScDataset

from ..models.extension import NrmlsBC
from .base import FunctionalBase


class scSingleBatch(FunctionalBase):
    """scSingleBatch model for scRNA-seq data.

    Attributes:
        model (TranscriptFormer): The model used for training.
    """

    def __init__(self, device=None, **kwargs):
        model = TranscriptFormer(**kwargs)
        super().__init__(model, device=device)

    def handle_input_tuple(self, input_tuple):
        X = input_tuple
        return self.loss(X.to(self.device))


class scMultiBatchNrmls(FunctionalBase):
    """scMultiBatchNrmls model for multi-batch scRNA-seq data.

    This class provides an interface for training and using the extension of the backbone model for multi-batch scRNA-seq data.

    Attributes:
        model (NrmlsBC): The model used for training.
        _factor (float): The beta factor for tuning the kl loss.
    """

    def __init__(self, num_batches=2, device=None, **kwargs):
        """Initialize the scMultiBatchNrmls model.

        Args:
            **kwargs: Additional keyword arguments for the model.

        """
        kwargs["dec_norm"] = kwargs.get("dec_norm", None)
        model = NrmlsBC(num_batches=num_batches, **kwargs)
        super().__init__(model=model, device=device)
        self.model: NrmlsBC
        self._trained = False
        self._num_batches = num_batches

    def _tune_loss(self, x, class_label, batch_rep):
        """Compute the tuning loss for the model in the semi-supervised setting.
        This method is used for calculating the loss for the model in the semi-supervised setting, where the scRNA-seq data is labeled.
        And only the labeled data is used to calculate the cross-entropy loss, and the KL divergence loss is calculated using the entire dataset.
        Reconstruction loss is not used in this setting.

        Args:
            x (torch.Tensor): Input data.
            class_label (torch.Tensor): Class labels.
            batch_rep (torch.Tensor): Batch representation.
            beta (float, optional): Beta value for tuning. Defaults to 0.01.

        Returns:
            dict: Dictionary containing the computed losses.
                - cl_loss (torch.Tensor): Classification loss.
                - kl_loss (torch.Tensor): KL divergence loss.
        """
        cls_rep = self.model.encode(x, batch_rep)
        clsf_rep = self.model.class_clsf_head(
            torch.cat([cls_rep, self.model.anchors], dim=0),
        )
        cl_loss = F.cross_entropy(
            clsf_rep,
            torch.cat(
                [class_label, torch.arange(
                    self.model.num_classes).to(self.device)]
            ),
        )
        if not self.model.readout.variational:
            return dict(
                cl_loss=cl_loss.mean(),
            )
        kl_loss = self.model.readout.kl_loss()
        return dict(
            cl_loss=cl_loss.mean(),
            kl_loss=self._factor * kl_loss.mean(),
        )

    def _loss(self, x, batch_rep):
        """Calculates the loss for the given input and batch representation.
        This method is used for calculating the loss for the model in the unsupervised setting.

        Args:
            x: The input data.
            batch_rep: The batch representation.

        Returns:
            The calculated loss.
        """
        rep_ts = self.model.encode_ts(x, batch_rep)
        decode_dict = self.model.decode_ts(rep_ts, batch_rep=batch_rep, x_gd=x)
        return self.loss_fn(**decode_dict)

    def loss(self, x, batch_rep, class_label=None,):
        """Calculates the loss function for the model.
        The internal loss function is selected based on the availability of class labels, for semi-supervised and unsupervised settings.

        Args:
            x (Tensor): The input data.
            batch_rep (Tensor): The batch representation.
            class_label (Tensor, optional): The class label (default: None).
            beta (float, optional): The beta parameter for tuning the loss (default: 1e-2).

        Returns:
            Tensor: The calculated loss value.
        """
        if class_label is None:
            return self._loss(x, batch_rep)
        return self._tune_loss(
            x=x,
            class_label=class_label,
            batch_rep=batch_rep,
        )

    def train(self):
        logger.info("use .train_batch instead")
        raise NotImplementedError

    def handle_input_tuple(self, input_tuple,):
        """Handles the input tuple and calculates the loss.
        This method is implemented to handle the input tuple and calculate the loss for `train_batch` method, see `train_batch` in parent class `FunctionalBase`.

        Args:
            input_tuple (tuple): The input tuple containing the data and labels.
            beta (float, optional): The beta value for the loss calculation. Defaults to 1e-2.

        Returns:
            dict: A dictionary containing the loss values.
        """
        if not self._trained:
            X, batch_label = input_tuple
            batch_rep = self.get_batch_ohenc(batch_label)
            loss_dict = self.loss(X.to(self.device), batch_rep.to(self.device))
        else:
            X, label, batch_label = input_tuple
            batch_rep = self.get_batch_ohenc(batch_label)
            loss_dict = self.loss(
                X.to(self.device),
                class_label=label.to(self.device),
                batch_rep=batch_rep.to(self.device),
            )
        return loss_dict

    @torch.no_grad()
    def _embed(self, dataset, tsfmr_out=False, as_numpy=True):
        """Embeds the dataset using the model.

        Args:
            dataset: The dataset to be embedded.
            tsfmr_out: Whether to use the transformer output for embedding.
            as_numpy: Whether to return the embedding as a numpy array.

        Returns:
            The embedded dataset.

        """
        self.model.eval()
        self.model._smooth = False
        self.model.to(self.device)
        prev_mode = dataset.set_mode("multi_batches")
        loaders = self.make_loaders(
            dataset, batch_size=512, split_rate=0, shuffle=False
        )
        dim = self.model.module_dim if not tsfmr_out else self.model.hidden_dim
        rep = torch.zeros(len(dataset), dim)
        offset = 0
        fn = self.model.encode if not tsfmr_out else self.model.encode_ts
        for x, batch_label in loaders[0]:
            batch_rep = self.get_batch_ohenc(batch_label, device=self.device)
            rep[offset: offset + len(x)] = fn(
                x.to(self.device), batch_rep=batch_rep
            ).cpu()
            offset += len(x)
        self.model.cpu()
        dataset.set_mode(mode=prev_mode)
        if as_numpy:
            return rep.numpy()
        return rep

    def embed(self, dataset, tsfmr_out=False, as_numpy=True):
        """Embeds the given dataset using the model.

        Args:
            dataset: The dataset to be embedded.
            tsfmr_out: Whether to return the output of the transformer layer. Defaults to False.
            as_numpy: Whether to return the embeddings as numpy arrays. Defaults to True.

        Returns:
            The embeddings of the dataset.
        """
        return self._embed(dataset, tsfmr_out, as_numpy)

    @torch.no_grad()
    def get_anchors(self):
        """
        Returns the anchors used by the model.

        Returns:
            numpy.ndarray: An array of anchors.
        """
        return self.model.anchors.cpu().numpy()

    def run(
        self,
        adata: AnnData,
        dataset: BaseDataset | MaskedDataset,
        epochs=400,
        batch_size: Optional[int] = 512,
        lr=1e-3,
        split_rate=0.2,
        tune_epochs=100,
        tune_lr=1e-4,
        need_anchors=True,
        unlabeled_key=None,
        groupby=None,
        key_added="X_rep",
        kl_cutoff=None,
        beta1=1e-2,
        beta2=1e-3,
        reset=False,
    ):
        """Run the function model.

        Args:
            adata (AnnData): The annotated data matrix.
            dataset (BaseDataset): The dataset object.
            epochs (int): The number of training epochs. Default is 1.
            batch_size (Optional[int]): The batch size. Default is None.
            lr (float): The learning rate. Default is 1e-3.
            split_rate (float): The split rate for train-test split. Default is 0.2.
            tune_epochs (int): The number of finetuning epochs. Default is 20.
            tune_lr (float): The learning rate for finetuning. Default is 1e-4.
            need_anchors (bool): Whether to use anchors. Default is True.
            unlabeled_key (Optional[str]): The key for unlabeled batch. Default is None.
            groupby (Optional[str]): The key for grouping. Default is None.
            key_added (str): The key for the added data. Default is 'X_rep'.
            kl_cutoff (Optional[float]): The cutoff for KL loss. Default is None.
            beta (float): The beta value for KL loss. Default is 1e-2.
        """

        super().run(
            adata,
            dataset,
            epochs,
            batch_size,
            split_rate,
            key_added,
            kl_cutoff=kl_cutoff,
            obs_key=dataset.batch_key,
            reset=reset,
            beta=beta1,
            lr=lr,
        )
        self._trained = True
        if need_anchors and dataset.class_key is None:
            logger.info("class_key not found")
        if need_anchors and dataset.class_key is not None:
            self.refine(
                adata=adata,
                dataset=dataset,
                epochs=tune_epochs,
                batch_size=batch_size,
                lr=lr,
                tune_lr=tune_lr,
                split_rate=split_rate,
                unlabeled_key=unlabeled_key,
                groupby=groupby,
                key_added="X_anchord",
                kl_cutoff=kl_cutoff,
                beta=beta2,
            )

    def refine(
        self,
        adata: AnnData,
        dataset: BaseDataset | MaskedDataset,
        epochs=1,
        batch_size: Optional[int] = None,
        lr=1e-3,
        split_rate=0.2,
        tune_lr=1e-4,
        unlabeled_key=None,
        groupby=None,
        key_added="X_anchord",
        kl_cutoff=None,
        beta=1e-2,
    ):
        """Refine the model when class labels are available.
        Using the class labels, i.e. the cell type information, the model is refined to improve the classification performance and the quality of the embeddings.

        Args:
            adata (AnnData): The annotated data matrix.
            dataset (BaseDataset): The dataset object.
            epochs (int): The number of training epochs. Default is 1.
            batch_size (Optional[int]): The batch size. Default is None.
            lr (float): The learning rate. Default is 1e-3.
            split_rate (float): The split rate for train-test split. Default is 0.2.
            tune_lr (float): The learning rate for finetuning. Default is 1e-4.
            unlabeled_key (Optional[str]): The key for unlabeled batch. Default is None.
            groupby (Optional[str]): The key for grouping. Default is None.
            key_added (str): The key for the added data. Default is 'X_anchord'.
            kl_cutoff (Optional[float]): The cutoff for KL loss. Default is None.
            beta (float): The beta value for KL loss. Default is 1e-2.
        """
        self.model.init_anchor(num_classes=dataset.num_classes)
        if tune_lr is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.module.parameters(), "lr": tune_lr},
                    {"params": self.model.moduler.parameters(), "lr": tune_lr},
                    {"params": self.model.expand.parameters(), "lr": tune_lr},
                    {"params": self.model.readout.parameters(), "lr": lr},
                    {"params": self.model.class_clsf_head.parameters(), "lr": lr},
                    {"params": self.model.anchors, "lr": lr},
                    {"params": self.model.cls_token, "lr": lr},
                ]
            )
        else:
            self.init_optimizer(lr=lr)
        self.lossconfig["kl_loss"]["patience"] = 1
        if unlabeled_key is not None:
            logger.info(f"Unlabeled batch: {unlabeled_key}")
            _dataset = dataset.subset(key=unlabeled_key, col=groupby)
        else:
            _dataset = dataset

        prev_mode = _dataset.set_mode()
        super().run(
            _dataset.adata,
            _dataset,
            epochs,
            batch_size,
            split_rate,
            kl_cutoff=kl_cutoff,
            key_added=key_added,
            obs_key=dataset.class_key,
        )
        dataset.set_mode(prev_mode)
        self.add_embed(adata, dataset=dataset, key_added="X_anchord")
        adata.uns["anchors"] = self.get_anchors()

    def _train_clsf(self, input_tuple):
        """Trains the classification model.

        Args:
            input_tuple (tuple): A tuple containing the classification representation,
                                class label, and additional information.

        Returns:
            dict: A dictionary containing the classification loss.
        """
        cls_rep, class_label, _ = input_tuple
        cls_rep = cls_rep.to(self.device)
        class_label = class_label.to(self.device)
        clsf_rep = self.model.class_clsf_head(cls_rep)
        cl_loss = F.cross_entropy(clsf_rep, class_label)
        return dict(
            cl_loss=cl_loss.mean(),
        )

    @torch.no_grad()
    def clsf(
        self,
        adata,
        use_rep,
        out: Literal["hard", "soft"] = "soft",
        key_added="pred_celltype",
        as_numpy=True,
    ):
        """Perform cell type classification on the input data.

        Args:
            adata (AnnData): Annotated data object.
            use_rep (str): Key of the representation in `adata.obsm` to be used for classification.
            out (str, optional): Type of output. Either 'hard' for hard classification or 'soft' for soft classification. Default is 'soft'.
            key_added (str, optional): Key to add to `adata.obs` for storing the predicted cell types. Default is 'pred_celltype'.
            as_numpy (bool, optional): Whether to return the predicted cell types as a numpy array. Default is True.

        Returns:
            pred (numpy.ndarray or torch.Tensor): Predicted cell types. If `as_numpy` is True, returns a numpy array. Otherwise, returns a torch.Tensor.
        """

        self.model.to(self.device)
        try:
            rep = adata.obsm[use_rep]
            pred = self.clsf_(torch.from_numpy(rep).to(self.device))
        except Exception:
            raise
        self.model.cpu()
        if as_numpy:
            pred = pred.numpy()
        if out == "hard":
            pred = pred.argmax(1)
            adata.obs[key_added] = pred
        if key_added is None or out == "soft":
            return pred

    @torch.no_grad()
    def clsf_(self, rep):
        """
        Perform classification using the given representation.

        Args:
            rep: The input representation.

        Returns:
            The softmax output of the classification head.
        """
        return F.softmax(self.model.class_clsf_head(rep.to(self.device))).cpu()

    @torch.no_grad()
    def impute(
        self,
        adata: AnnData | None,
        dataset: ScDataset,
        decode_dict=None,
        key_added: str | None = "expected_counts",
        rep_key: str | None = None,
        layer_key=None,
        rep: torch.Tensor | None = None,
        qc=False,
        **kwargs,
    ):
        """Imputes missing values in the given dataset using the trained model.

        Args:
            adata (AnnData): Annotated data object containing the dataset.
            dataset (BaseDataset): Dataset object containing the gene expression data.
            decode_dict (dict, optional): Dictionary containing the decoding information. If not provided, it will be generated using the trained model.
            key_added (str, optional): Key to store the imputed values in the `adata.layers` attribute. Default is 'expected_counts'.
            rep_key (str, optional): Key to retrieve the representation matrix from `adata.obsm` attribute. If not provided, it will be inferred from the `adata.obsm` keys.
            layer_key (str, optional): Key to retrieve the layer-specific gene expression data from `adata.layers` attribute.
            rep (torch.Tensor, optional): Representation matrix. If not provided, it will be generated using the `_embed` method.
            qc (bool, optional): Flag indicating whether to perform quality control on the imputed values. Default is False.
            **kwargs: Additional keyword arguments to be passed to the `super().impute` method.

        Returns:
            res: The result of the `super().impute` method.

        """
        if decode_dict is None:
            if rep is None and (rep_key is None or rep_key not in adata.obsm.keys()):
                rep = self._embed(dataset, as_numpy=False)
            elif adata is not None and rep_key in adata.obsm_keys():
                rep = rep if rep is not None else torch.from_numpy(
                    adata.obsm[rep_key])
            assert rep is not None

            self.model.to(self.device)
            decode_dict = self.model.decode(
                cls_rep=rep.to(self.device),
                x_gd=dataset.gene_expr.to(self.device),
                batch_rep=self.get_batch_ohenc(
                    dataset.batch_label).to(self.device),
            )
        res = super().impute(
            adata=adata,
            dataset=dataset,
            decode_dict=decode_dict,
            key_added=key_added,
            rep_key=rep_key,
            layer_key=layer_key,
            rep=rep,
            x=None,
            qc=qc,
            **kwargs,
        )
        self.model.cpu()
        return res

    def reset_model(self):
        super().reset_model()
        self._trained = False
