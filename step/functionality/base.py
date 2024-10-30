import torch
import torch.utils
from anndata import AnnData

from step.manager import logger
from step.models.transcriptformer import TranscriptFormer
from step.utils.dataset import BaseDataset, MaskedDataset

from .comps.model_ops import ModelOps
from .comps.trainer import Trainer


class FunctionalBase(ModelOps, Trainer):
    """
    FunctionalBase is a compound class that combines the ModelOps and Trainer classes.

    Attributes:
        model (TranscriptFormer): The model to be trained.
        split_rate (float): The ratio of validation data to the total data.
        use_earlystop (bool): A flag indicating whether to use early stopping during training.
        device (str): The device to use for training. Defaults to "cuda" if a GPU is available, otherwise "cpu".
    """

    def __init__(self, model: TranscriptFormer, use_earlystop=True, device=None):
        """Initialize the FunctionalBase.

        Args:
            model (TranscriptFormer): The model to be wrapped.
            use_earlystop (bool, optional): A flag indicating whether to use early stopping during training. Defaults to True.
        """
        ModelOps.__init__(self, model=model, device=device)
        Trainer.__init__(self, model=model)
        self.split_rate = 0.0
        self.use_earlystop = use_earlystop
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(
        self,
        adata: AnnData,
        dataset: BaseDataset | MaskedDataset,
        epochs=1,
        batch_size: int | None = None,
        split_rate=0.2,
        key_added="X_rep",
        obs_key=None,
        call_func=None,
        kl_cutoff=None,
        reset=False,
        beta=0.01,
        lr=1e-3,
    ):
        """Run the training and embedding process.

        Args:
            adata (AnnData): Annotated data object.
            dataset (Union[ScDataset, StDataset]): Dataset object containing gene expression data.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Batch size for training. Defaults to None.
            split_rate (float, optional): Split rate for train-test split. Defaults to 0.2.
            key_added (str, optional): Key to store the embedding in `adata.obsm`. Defaults to 'X_rep'.
            obs_key (str, optional): Key in `adata.obs` to use for train-test split. Defaults to None.
            call_func (callable, optional): Callback function to be called after each epoch. Defaults to None.
            kl_cutoff (float, optional): KL divergence cutoff value. Defaults to None.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        if reset:
            logger.warning("Resetting model")
            self.reset_model()
        self.init_optimizer(lr=lr)
        self.split_rate = split_rate
        self.model.to(self.device)
        self.set_kl_cutoff(kl_cutoff)
        self.set_beta(beta)
        if batch_size is not None:
            loaders = self.make_loaders(
                dataset=dataset,
                batch_size=batch_size,
                split_rate=split_rate,
                shuffle=True,
                obs_key=obs_key,
            )
            self.train_batch(
                epochs=epochs, loaders=loaders, call_func=call_func  # type:ignore
            )
        else:
            self.train(
                epochs=epochs, X=dataset.gene_expr, call_func=call_func
            )  # type:ignore
        torch.cuda.empty_cache()
        self.add_embed(adata, dataset=dataset, key_added=key_added)
        torch.cuda.empty_cache()
        self.model.cpu()

    def load_checkpoint(self, checkpoint):
        """Loads a checkpoint file and updates the model's state dictionary.

        Args:
            checkpoint (str): The path to the checkpoint file.

        Returns:
            None
        """
        logger.info("Loading backbone model...")
        state_dict = torch.load(checkpoint)
        if 'anchors' in state_dict:
            logger.info("Loading anchors...")
            getattr(self.model, 'init_anchor')(state_dict['anchors'].shape[0])
        import re
        smoother_keys = [key for key in state_dict.keys() if re.match(r'^smoother', key)]
        if smoother_keys:
            self.model.init_smoother_with_builtin()
        if 'px_r' in state_dict:
            state_dict["decoder.px_r"] = state_dict.pop("px_r")
        if 'l_scale' in state_dict:
            state_dict["decoder.l_scale"] = state_dict.pop("l_scale")
        self.model.load_state_dict(state_dict)
        logger.info("Backbone model loaded.")
