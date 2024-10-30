import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from step.manager import logger
from step.models.transcriptformer import TranscriptFormer
from step.utils.dataset import BaseDataset, MaskedDataset


def train_loop_formatter(train_func):
    """
    A decorator function that formats the training loop.

    Args:
        train_func (function): The training function to be decorated.

    Returns:
        function: The decorated training function.

    """

    def wrapper(ref, epochs: int, writer=None, **kwargs):
        state: Dict[str, Any] = {"is_init": False}
        state["epochs"] = epochs
        flag = None
        if epochs == 1:
            epoch = epochs
            state["epoch"] = [f"{epoch+1}/{epochs}"]
            state["epoch_val"] = epoch + 1
            state["last_epoch"] = (epoch + 1) == epochs
            state["start_time"] = time.time()
            state["writer"] = writer
            flag = train_func(ref, state=state, **kwargs)
        else:
            with tqdm(range(epochs), unit="epoch") as tepochs:
                for epoch in tepochs:
                    state["epoch"] = [f"{epoch+1}/{epochs}"]
                    state["epoch_val"] = epoch + 1
                    state["last_epoch"] = (epoch + 1) == epochs
                    state["start_time"] = time.time()
                    state["writer"] = writer

                    try:
                        flag = train_func(ref, state=state,
                                          ** kwargs)  # type:ignore
                        tepochs.set_postfix(**state.get("loss_dict", {}))
                    except Exception as e:
                        logger.error(state)
                        logger.error(ref.early_stopping)
                        raise e
                    if flag is not None and flag:
                        logger.info("Early Stopping triggered")
                        for v in ref.early_stopping.values():
                            v.trace_func(
                                f"EarlyStopping counter: {v.counter} out of {v.patience}"
                            )
                        break
        if writer is not None:
            writer.close()
        ref.early_stopping.clear()
        state = {"is_init": False}

    return wrapper


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.early_stop = False


class Trainer(object):
    """
    The Trainer class is responsible for training the TranscriptFormer model.

    Attributes:
        model (TranscriptFormer): The TranscriptFormer model being trained.
        optimizer (torch.optim.Adam): The optimizer used for model parameter updates.
        lr_scheduler (torch.optim.lr_scheduler.MultiStepLR): The learning rate scheduler.
        early_stopping (Dict[str, EarlyStopping]): A dictionary of EarlyStopping objects for each loss type.
        lossconfig (dict): A dictionary containing the configuration for different loss types.
        defualt_lcfg (dict): The default configuration for loss types.

    """

    def __init__(self, model: TranscriptFormer):
        """
        Initializes the Trainer object.

        Args:
            model (TranscriptFormer): The TranscriptFormer model being trained.
        """
        self.model = model
        self.init_optimizer()
        self.set_lr_scheduler()
        self.model.train()
        self.early_stopping: Dict[str, EarlyStopping] = {}
        # self.lr_scheduler = kwargs.get('lr_scheduler', defualt_lr_scheduler)
        self.lossconfig = {
            "recon_loss": {
                "patience": 30,
                "delta": 1e-2,
            },
            "kl_loss": {
                "patience": 10,
                "delta": 1e-3,
            },
            "cl_loss": {
                "patience": 20,
                "delta": 1e-4,
            },
            "gate_loss": {
                "patience": 15,
                "delta": 1e-3,
            },
        }
        self.defualt_lcfg = {
            "patience": 15,
            "delta": 1e-3,
        }

    def mode_switch_to_train(self):
        if not self.model.training:
            self.model.train()

    def init_optimizer(
        self,
        lr=1e-3,
        tune_lr=1e-4,
        stage=1,
        tune_batch_embedding=True,
    ):
        """
        Initializes the optimizer for model parameter updates.

        Args:
            lr (float): The learning rate for the base model parameters.
            tune_lr (float): The learning rate for the fine-tuned model parameters.
            stage (int): The training stage.
            tune_batch_embedding (bool): Whether to tune the batch embedding.
        """
        if stage == 1:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(
                [
                    {"params": self.model.smoother.parameters(), "lr": lr},
                    {"params": self.model.module.parameters(), "lr": tune_lr},
                    {"params": self.model.moduler.parameters(), "lr": tune_lr},
                    {"params": self.model.expand.parameters(), "lr": tune_lr},
                    {"params": self.model.readout.parameters(), "lr": tune_lr},
                    {"params": self.model.decoder.parameters(), "lr": tune_lr},
                    {"params": self.model.cls_token, "lr": tune_lr},
                ]
            )
            if getattr(self, "_num_batches", 1) > 1 and tune_batch_embedding:
                self.optimizer.add_param_group(
                    {
                        "params": self.model.batch_readout.parameters(),
                        "lr": tune_lr,
                    }
                )
                self.optimizer.add_param_group(
                    {
                        "params": self.model.batch_embedding,
                        "lr": tune_lr,
                    }
                )

    def set_lr_scheduler(self, lr_scheduler=None):
        """
        Sets the learning rate scheduler.

        Args:
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            gamma (float): The multiplicative factor of learning rate decay.
            *milestones: The list of epoch indices at which to decay the learning rate.
        """
        # logger.info("Setting LR scheduler")
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        pass

    def update_model_params(self, loss_dict, epoch):
        """
        Updates the model parameters based on the loss.

        Args:
            loss_dict (dict): A dictionary containing the loss values.
        """
        loss = sum([v for _, v in loss_dict.items()
                   if isinstance(v, torch.Tensor)])
        self.optimizer.zero_grad()
        loss.backward()  # type:ignore
        self.optimizer.step()
        # self.lr_scheduler.step()

    def setup_early_stopping(self, loss_dict: dict):
        """
        Sets up the EarlyStopping objects for each loss type.

        Args:
            loss_dict (dict): A dictionary containing the loss values.
        """
        self.early_stopping.clear()
        for key in loss_dict.keys():
            self.early_stopping[f"val_{key}"] = EarlyStopping(
                **self.lossconfig.get(key, self.defualt_lcfg)
            )

    def cum_loss(
        self,
        loss_dict: dict,
        total_loss_dict: dict,
        use_earlystop=False,
        n_batch: int = -1,
        n_epoch: int = -1,
    ):
        """
        Computes the cumulative loss.

        Args:
            loss_dict (dict): A dictionary containing the loss values.
            total_loss_dict (dict): A dictionary containing the cumulative loss values.
            use_earlystop (bool): Whether to use early stopping.
            n_batch (int): The current batch number.
            n_epoch (int): The current epoch number.

        Returns:
            dict: The updated total_loss_dict.
        """
        # offest of n is 0
        if n_epoch == 1 and use_earlystop:
            self.early_stopping = {}
            self.setup_early_stopping(loss_dict)
        if n_batch == 0:
            return loss_dict
        for key in loss_dict.keys():
            total_loss_dict[key] = total_loss_dict[key] * (
                n_batch / (n_batch + 1)
            ) + loss_dict[key] / (n_batch + 1)
        return total_loss_dict

    def check_early_stop(self, total_loss_dict: dict):
        """
        Checks if early stopping criteria are met.

        Args:
            total_loss_dict (dict): A dictionary containing the cumulative loss values.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        flag = True
        for key in self.early_stopping.keys():
            self.early_stopping[key](total_loss_dict[key], self.model)
            flag &= self.early_stopping[key].early_stop
        # if flag:
        return flag

    def format_loss_dict(
        self,
        loss_dict: dict,
        state: dict,
        valid_loss_dict: Optional[dict] = None,
        endepoch: bool = False,
    ):
        """
        Formats the loss dictionary for display.

        Args:
            loss_dict (dict): A dictionary containing the loss values.
            state (dict): A dictionary containing the current state.
            valid_loss_dict (dict, optional): A dictionary containing the validation loss values.
            endepoch (bool, optional): Whether it is the end of an epoch.
        """
        # clear_output(wait=True)
        if valid_loss_dict is not None:
            _valid_loss_dict = {
                k: v for k, v in valid_loss_dict.items() if isinstance(v, torch.Tensor)
            }
        elif self.split_rate > 1e-2:
            _valid_loss_dict = {
                f"val_{k}": "-"
                for k, v in loss_dict.items()
                if isinstance(v, torch.Tensor)
            }
        else:
            _valid_loss_dict = {}
        loss_dict = {**loss_dict, **_valid_loss_dict}

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                str_fmt = f"{v.item():.3f}"
                loss_dict[k] = str_fmt if len(
                    str_fmt) <= 8 else f"{v.item():.2e}"
            elif isinstance(v, str):
                loss_dict[k] = v
            else:
                loss_dict[k] = f"{v:.2e}"

        for key in _valid_loss_dict.keys():
            if state["epoch_val"] != 1:
                v = loss_dict[key]
                if (
                    self.early_stopping
                    and self.early_stopping[key].best_score is not None
                ):
                    # type:ignore
                    best_v = -1 * self.early_stopping[key].best_score
                    str_fmt = (
                        f"{best_v:.3f}" if abs(
                            best_v) >= 1e-3 else f"{best_v:.2e}"
                    )
                    best_v = str_fmt if len(str_fmt) <= 8 else f"{best_v:.2e}"
                    loss_dict[key] = f"{v}/{best_v}"
            else:
                v = loss_dict[key]
                loss_dict[key] = f"{v}/-"
        state["loss_dict"] = loss_dict

    def split_X(self, X, split_rate):
        """Splits the input data into training and validation sets based on the given split rate.

        Args:
            X (array-like): The input data to be split.
            split_rate (float): The ratio of validation data to the total data.

        Returns:
            tuple: A tuple containing the training data and validation data.
        """
        if split_rate < 0.1:
            valid_X = None
            train_X = X
            return train_X, valid_X
        train_X, valid_X = train_test_split(
            X, test_size=split_rate, shuffle=True)
        return train_X, valid_X

    def handle_input_tuple(self, input_tuple):
        """Handles the input tuple and calculates the loss.

        Args:
            input_tuple (tuple): A tuple containing the input data and the target data.

        Returns:
            dict: A dictionary containing the loss values.

        """
        X, _ = input_tuple
        loss_dict = self.loss(X.to(self.device))
        return loss_dict

    def handle_ginput_tuple(self, input_tuple, ind, step):
        return {}

    @train_loop_formatter
    def train_batch(self, state, loaders, call_func: Optional[Callable] = None):
        """
        Trains the model on a batch of data.

        Args:
            state (dict): The state dictionary containing training parameters.
            loaders (tuple): A tuple containing the train and validation loaders.
            call_func (Callable, optional): The function to call on the input tuple. Defaults to None.

        Returns:
            bool: A flag indicating whether to stop training early.
        """
        self.model.train()
        if call_func is None:
            call_func = self.handle_input_tuple
        train_loader, valid_loader = loaders
        writer = state.get("writer", None)
        n_steps = len(train_loader)
        train_loss_dict = {}
        for i, input_tuple in enumerate(train_loader):
            loss_dict = call_func(input_tuple)
            self.update_model_params(loss_dict, state['epoch_val'] - 1)
            train_loss_dict = self.cum_loss(
                loss_dict,
                train_loss_dict,
                use_earlystop=self.use_earlystop,
                n_epoch=state["epoch_val"],
                n_batch=i,
            )
            if writer is not None:
                for k, v in train_loss_dict.items():
                    writer.add_scalar(
                        k, v.item(), (state["epoch_val"] - 1) * n_steps + i + 1
                    )
            self.format_loss_dict(loss_dict, state)
            del loss_dict
        if valid_loader is not None:
            state["validation"] = True
            valid_loss_dict = self.validate_loader(valid_loader, call_func)
            self.format_loss_dict(
                loss_dict=train_loss_dict,
                valid_loss_dict=valid_loss_dict,
                state=state,
                endepoch=True,
            )
            flag = False
            if self.use_earlystop:
                flag = self.check_early_stop(valid_loss_dict)  # type:ignore
            del train_loss_dict, valid_loss_dict
            return flag
        del train_loss_dict

    @train_loop_formatter
    def train(
        self,
        state,
        X,
        train_ind=None,
        valid_ind=None,
        call_func: Optional[Callable] = None,
        **kwargs,
    ):
        """Train the model using the provided data.

        Args:
            state (dict): The state dictionary containing training parameters.
            X (torch.Tensor): The input data.
            train_ind (Optional[List[int]]): The indices of the training data.
            valid_ind (Optional[List[int]]): The indices of the validation data.
            call_func (Optional[Callable]): The function to calculate the loss.
            **kwargs: Additional keyword arguments to be passed to the loss function.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """

        self.model.train()
        # self.set_warmup(epochs=state['epochs'])
        if call_func is None:
            call_func = self.loss
        loss_dict = call_func(X.to(self.device), ind=train_ind, **kwargs)
        self.update_model_params(loss_dict, state["epoch_val"] - 1)  # type:ignore
        self.format_loss_dict(loss_dict, state)
        if valid_ind is not None:
            state["validation"] = True
            valid_loss_dict = self.validate_X(
                X, ind=valid_ind, call_func=call_func)
            self.format_loss_dict(
                loss_dict=loss_dict,
                valid_loss_dict=valid_loss_dict,
                state=state,
                endepoch=True,
            )
            flag = False
            if self.use_earlystop:
                if state["epoch_val"] == 1:
                    self.setup_early_stopping(loss_dict)
                flag = self.check_early_stop(valid_loss_dict)  # type:ignore
            del loss_dict, valid_loss_dict
            return flag
        del loss_dict

    @train_loop_formatter
    def train_node_sampler(self, state, gloader, dataset, train_ind=None, valid_ind=None):
        """Trains the model with node sampler.

        Args:
            state (dict): The state dictionary containing training parameters.
            gloader (DataLoader): The data loader for the training data.
            train_ind (list, optional): The indices of the training data. Defaults to None.
            valid_ind (list, optional): The indices of the validation data. Defaults to None.
        """
        self.model.train()
        # self.set_warmup(epochs=state['epochs'], loader=gloader)
        n_steps = len(gloader)
        writer = state.get("writer", None)
        with tqdm(enumerate(gloader), unit="step", total=n_steps) as tgloader:
            for i, ginput_tuple in tgloader:
                cur_step = i + 1
                loss_dict = self.handle_ginput_tuple(
                    ginput_tuple, dataset, train_ind, step=cur_step
                )
                graph_ids = loss_dict.pop("graph_ids", None)
                self.update_model_params(loss_dict, i)
                if writer is not None:
                    for k, v in loss_dict.items():
                        writer.add_scalar(k, v.item(), cur_step)
                self.format_loss_dict(loss_dict, state)
                info = state.get("loss_dict", {})
                info["graph_ids"] = graph_ids
                tgloader.set_postfix(info)
                del loss_dict

    @train_loop_formatter
    def train_graph_batch(self, state, gloader, dataset, train_ind=None, valid_ind=None):
        """Trains the model on a batch of graph inputs.

        Args:
            state (dict): The state dictionary containing training parameters.
            gloader (DataLoader): The data loader for loading graph inputs.
            train_ind (list, optional): The indices of the training samples. Defaults to None.
            valid_ind (list, optional): The indices of the validation samples. Defaults to None.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """

        self.model.train()
        # self.set_warmup(epochs=state['epochs'], loader=gloader)
        train_loss_dict = {}
        n_steps = len(gloader)
        writer = state.get("writer", None)
        for i, ginput_tuple in enumerate(gloader):
            loss_dict = self.handle_ginput_tuple(ginput_tuple, dataset=dataset, ind=train_ind)
            loss_dict.pop("graph_ids", None)
            self.update_model_params(loss_dict, epoch=state['epoch_val'] - 1)
            train_loss_dict = self.cum_loss(
                loss_dict,
                train_loss_dict,
                use_earlystop=self.use_earlystop,
                n_epoch=state["epoch_val"],
                n_batch=i,
            )
            if writer is not None:
                for k, v in train_loss_dict.items():
                    writer.add_scalar(
                        k, v.item(), (state["epoch_val"] - 1) * n_steps + i + 1
                    )
            self.format_loss_dict(loss_dict, state)
            del loss_dict
        if valid_ind is not None:
            state["validation"] = True
            valid_loss_dict = {}
            for i, ginput_tuple in enumerate(gloader):
                loss_dict = self.handle_ginput_tuple(ginput_tuple, valid_ind)
                valid_loss_dict = self.cum_loss(
                    loss_dict,
                    valid_loss_dict,
                    use_earlystop=self.use_earlystop,
                    n_epoch=state["epoch_val"],
                    n_batch=i,
                )
            self.format_loss_dict(
                loss_dict=train_loss_dict,
                valid_loss_dict=valid_loss_dict,
                state=state,
                endepoch=True,
            )
            flag = False
            if self.use_earlystop:
                flag = self.check_early_stop(valid_loss_dict)  # type:ignore
            del train_loss_dict, valid_loss_dict
            return flag
        del train_loss_dict

    @torch.no_grad()
    def validate_loader(self, loader, call_func: Optional[Callable] = None):
        """Validates the given loader by evaluating the model on the input data.

        Args:
            loader (Iterable): The data loader containing the input data.
            call_func (Callable, optional): The function to be called for processing each input tuple.
                If not provided, self.handle_input_tuple will be used.

        Returns:
            dict: A dictionary containing the cumulative loss values for each metric.
                The keys are in the format 'val_{metric_name}'.
        """
        self.model.eval()
        if call_func is None:
            call_func = self.handle_input_tuple
        cum_loss_dict = {}
        for (
            i,
            input_tuple,
        ) in enumerate(loader):
            loss_dict = call_func(input_tuple)
            cum_loss_dict = self.cum_loss(
                loss_dict,
                cum_loss_dict,
                n_batch=i,
            )
        cum_loss_dict = {f"val_{k}": v for k, v in cum_loss_dict.items()}
        return cum_loss_dict

    @torch.no_grad()
    def validate_X(self, X, ind=None, call_func: Optional[Callable] = None):
        """Validates the input data X using the specified call_func.

        Args:
            X: The input data to be validated.
            ind: The index of the data.
            call_func: The function used to calculate the loss. If not provided, self.loss will be used.

        Returns:
            A dictionary containing the validation loss values.

        """
        self.model.eval()
        if call_func is None:
            call_func = self.loss
        total_loss_dict = call_func(X.to(self.device), ind=ind)
        total_loss_dict = {f"val_{k}": v for k, v in total_loss_dict.items()}
        return total_loss_dict

    @torch.no_grad()
    def validate_gloader(self, gloader):
        """Validates the given data loader by evaluating the model on the input data.

        Args:
            gloader (DataLoader): The data loader containing the input data.

        Returns:
            dict: A dictionary containing the cumulative loss values for each metric.
        """
        self.model.eval()
        cum_loss_dict = {}
        for i, ginput_tuple in enumerate(gloader):
            loss_dict = self.handle_ginput_tuple(ginput_tuple)
            cum_loss_dict = self.cum_loss(
                loss_dict,
                cum_loss_dict,
                n_batch=i,
            )
        cum_loss_dict = {f"val_{k}": v for k, v in cum_loss_dict.items()}
        return cum_loss_dict

    @staticmethod
    def make_loaders(
        dataset: BaseDataset | MaskedDataset,
        batch_size: int,
        split_rate: float = 0.2,
        shuffle: bool = True,
        obs_key: str | None = None,
        **kwargs,
    ):
        """Create data loaders for training and validation datasets.

        Args:
            dataset (Union[ScDataset, StDataset]): The dataset to create loaders for.
            batch_size (int): The batch size for the loaders.
            split_rate (float, optional): The ratio to split the dataset into training and validation sets. Defaults to 0.2.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            obs_key (Optional[str], optional): The key for the column in `adata.obs`. Defaults to None.

        Returns:
            Tuple[DataLoader, DataLoader]: The training and validation data loaders.
        """

        train_loader, valid_loader = None, None
        train_ds = dataset
        if (class_key := getattr(dataset, "class_key", None)) is not None and class_key == obs_key:
            logger.info(f"Found class key: {class_key}, setting mode to multi_batches_with_ct")
            mode = "multi_batches_with_ct"
        else:
            mode = dataset.mode
        if split_rate > 0.0:
            if obs_key is not None and not dataset.adata.obs[obs_key].isna().any():
                logger.info("Performing category random split")
                train_ds, valid_ds = category_random_split(
                    dataset, split_rate, obs_key=obs_key
                )
                train_ds.set_mode(mode)
                valid_ds.set_mode(mode)
            else:
                logger.info("Performing global random split")
                train_size = int((1 - split_rate) * len(dataset))
                valid_size = len(dataset) - train_size
                dataset.set_mode(mode)
                train_ds, valid_ds = random_split(
                    dataset, [train_size, valid_size])
            valid_loader = DataLoader(
                valid_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return (train_loader, valid_loader)


def category_random_split(
    dataset: Union[BaseDataset, MaskedDataset],
    split_rate: float = 0.2,
    obs_key: Optional[str] = None,
) -> Tuple[MaskedDataset, MaskedDataset]:
    """Splits the given dataset into training and validation datasets based on categories.

    Args:
        dataset (ScDataset): The input dataset to be split.
        split_rate (float, optional): The ratio of validation data to total data. Defaults to 0.2.
        obs_key (str, optional): The key of the column in `adata.obs` containing categories. Defaults to None.

    Returns:
        Tuple[ScDataset, ScDataset]: A tuple containing the training dataset and validation dataset.
    """

    train_indicies = []
    assert obs_key in dataset.adata.obs_keys()
    categories = dataset.adata.obs[obs_key].astype("category")
    all_indicies = np.arange(len(dataset))

    for label in categories.cat.categories:
        indicies = dataset.adata.obs[obs_key] == label
        label_ind = all_indicies[indicies.values]
        indicies = np.random.permutation(label_ind)
        train_size = int((1 - split_rate) * len(indicies))
        logger.info(f"Training size for {label}: {train_size}")
        train_indicies.extend(label_ind[:train_size])

    train_indicies = dataset.adata.obs.index.to_numpy()[train_indicies]
    dataset.adata.obs["train"] = False
    dataset.adata.obs.loc[train_indicies, "train"] = True
    train_ds = dataset.subset(col="train", exclude=False)
    valid_ds = dataset.subset(col="train", exclude=True)
    logger.info(f"train size: {len(train_ds)}")
    logger.info(f"valid size: {len(valid_ds)}")
    return train_ds, valid_ds
