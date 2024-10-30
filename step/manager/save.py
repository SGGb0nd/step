import json
import os
from pathlib import Path

import torch
from anndata import AnnData

from step.functionality.base import FunctionalBase
from step.manager import logger
from step.utils.dataset import BaseDataset


class Saver:
    """
    Save the model and the dataset.

    Attributes:
        funcmodel (FunctionalBase): The functional model.
        adata (AnnData): The AnnData object.
        dataset (BaseDataset): The dataset object.
    """

    def __init__(self, class_name, path: str | Path = '.'):
        """Initialize the Saver.

        Args:
            path (str | Path): The path to save the model and the dataset.

        """
        self.class_name = class_name
        self.path = path
        self.save_adata = False

    def save(self, funcmodel: FunctionalBase, adata: AnnData, dataset: BaseDataset):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        logger.info("Saving model...")
        torch.save(funcmodel.model.state_dict(),
                   os.path.join(self.path, "model.pth"))

        if hasattr(funcmodel, "st_decoder"):
            torch.save(funcmodel.st_decoder.state_dict(),
                       os.path.join(self.path, "st_decoder.pth"))

        if has_mixer := hasattr(funcmodel, "mixer"):
            torch.save(funcmodel.mixer.state_dict(),
                       os.path.join(self.path, "mixer.pth"))

        config = {}
        with open(os.path.join(self.path, "config.json"), "w") as f:
            logger.info("Saving model config...")
            config["class_name"] = self.class_name
            config["save_adata"] = self.save_adata
            config["model_config"] = funcmodel.model.args
            if hasattr(funcmodel.model, "smoother"):
                config["model_config"]["n_glayers"] = funcmodel.model.gargs['n_layers']
            if has_mixer:
                config['model_config']['mixer'] = funcmodel.mixer.args

            logger.info("Saving dataset config...")
            ds_config = {
                k: getattr(dataset, k)
                for k in dataset.process_fields
                if k not in ['adata', "is_human"]
            }
            ds_config["received_layer_key"] = dataset.received_layer_key
            config["dataset_config"] = ds_config
            if geneset := ds_config.pop('geneset', None) is not None:
                with open(os.path.join(self.path,  "geneset.txt"), 'w') as f:
                    f.writelines(geneset)

            if self.save_adata:
                logger.info("Saving adata...")
                adata.write_h5ad(os.path.join(self.path, "adata.h5ad"))
                config["adata_path"] = os.path.join(self.path, "adata.h5ad")

            json.dump(config, f, indent=4)

        return config

    @classmethod
    def get_instance(cls, class_name, path='.'):
        if not hasattr(cls, "_instance"):
            cls._instance = cls(class_name, path)
        return cls._instance
