import os
from collections import deque

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class FixedFileNumModelCheckpoint(ModelCheckpoint):
    def __init__(self, max_file_num: int, *args, **kwargs):
        """Constructor of FixedFileNumModelCheckpoint class. For details see ht
        tps://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.
        ModelCheckpoint.html.

        Args:
            max_file_num (int): Specify the maximum number of files that can be saved.
        """
        super().__init__(*args, **kwargs)
        self.max_file_num = max_file_num
        self.file_paths = deque()

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        self.file_paths.append(filepath)
        if len(self.file_paths) > self.max_file_num:
            file_to_delete = self.file_paths.popleft()
            os.remove(file_to_delete)
        super()._save_checkpoint(trainer, filepath)
