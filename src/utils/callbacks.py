import os
from collections import deque

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class FixedFileNumModelCheckpoint(ModelCheckpoint):
    def __init__(self, max_file_num: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_file_num = max_file_num
        self.file_paths = deque()

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if len(self.file_paths) >= self.max_file_num:
            file_to_delete = self.file_paths.pop(0)
            os.remove(file_to_delete)
        self.file_paths.append(filepath)
        super()._save_checkpoint(trainer, filepath)
