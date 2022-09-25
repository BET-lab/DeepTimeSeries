import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .chunk import BaseChunkSpec, ChunkExtractor
from .plotting import plot_chunks


class TimeSeriesDataset(Dataset):
    def __init__(self,
        data_frames: pd.DataFrame | list[pd.DataFrame],
        chunk_specs: list[BaseChunkSpec],
        return_time_index: bool = False,
    ):
        if isinstance(data_frames, pd.DataFrame):
            data_frames = [data_frames]
        self.data_frames = data_frames

        # Make chunk_specs from encoding, decoding and label specs.
        self.chunk_specs = chunk_specs

        self.return_time_index = return_time_index

        self._preprocess()

    def _preprocess(self):
        self.chunk_extractors = [
            ChunkExtractor(df, self.chunk_specs) for df in self.data_frames
        ]

        self.lengths = [
            len(df) - self.chunk_extractors[0].chunk_length + 1
            for df in self.data_frames
        ]

        self.min_start_time_index = \
            max(0, -self.chunk_extractors[0].chunk_min_t)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, i):
        cumsum = np.cumsum([0] + self.lengths)
        df_index = np.argmax(cumsum - i > 0) - 1

        chunk_extractor = self.chunk_extractors[df_index]
        start_time_index = i - cumsum[df_index] + self.min_start_time_index

        chunk_dict = chunk_extractor.extract(
            start_time_index, self.return_time_index
        )

        return chunk_dict

    def plot_chunks(self):
        plot_chunks(self.chunk_specs)