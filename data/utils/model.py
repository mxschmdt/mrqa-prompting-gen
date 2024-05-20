import logging
import os

import torch
from transformers import WEIGHTS_NAME


class SaveLoadModelMixin:
    def save(self: torch.nn.Module, output_dir: str, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        logging.info(f"Model saved to {output_dir}.")

    def load(self: torch.nn.Module, path: str):
        state_dict = torch.load(
            os.path.join(path, WEIGHTS_NAME), map_location=torch.device("cpu")
        )
        # removing unnecessary keys which can happen due to reducing model after a checkpoint has been saved
        used_weights = state_dict.keys() & self.state_dict().keys()
        missing_keys = set(self.state_dict().keys()).difference(state_dict.keys())
        if missing_keys:
            logging.warning(
                "The following weights cannot be loaded from the checkpoint: %s",
                ", ".join(missing_keys),
            )
        if used_weights != state_dict.keys():
            logging.warning(
                "Omitting the following weights on loading state dict: %s",
                ", ".join(state_dict.keys() - used_weights),
            )
            state_dict = {k: state_dict[k] for k in used_weights}
        self.load_state_dict(state_dict, strict=True)
        logging.info(f"Model loaded from {path}.")
