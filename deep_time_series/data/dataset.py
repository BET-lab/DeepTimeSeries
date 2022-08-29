import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .chunk import ChunkExtractor
from ..plotting import plot_chunks


def _merge_data_frames(
    data_frames: pd.DataFrame | list[pd.DataFrame]
):
    if isinstance(data_frames, pd.DataFrame):
        data_frames = [data_frames]

    data_frames = [df.copy() for df in data_frames]

    for i, df in enumerate(data_frames):
        if '__time_index' in df.columns:
            raise ValueError(
                f'"__time_index" column exists in data_frames[{i}]'
            )

        if '__time_series_id' in df.columns:
            raise ValueError(
                f'"__time_series_id" column exists in data_frames[{i}]'
            )

        df['__time_index'] = np.arange(len(df.index))
        df['__time_series_id'] = i

    return pd.concat(data_frames).reset_index(drop=True)


class TimeSeriesDataset(Dataset):
    def __init__(self,
        data_frames: pd.DataFrame | list[pd.DataFrame],
        chunk_specs,
        column_transformer,
        fit_column_transformer=True,
        return_time_index=False,
    ):
        self.data_frames = _merge_data_frames(data_frames=data_frames)
        # Make chunk_specs from encoding, decoding and label specs.
        self.chunk_specs = chunk_specs

        self.column_transformer = column_transformer
        self.fit_column_transformer = fit_column_transformer
        self.return_time_index = return_time_index

        self._preprocess()

    def _preprocess(self):
        self.data_frames.sort_values(by='__time_index', inplace=True)

        if self.fit_column_transformer:
            self.column_transformer.fit(self.data_frames)

        self.scaled_df = self.column_transformer.transform(self.data_frames)

        splitted_dfs = [
            df for _, df in self.scaled_df.groupby('__time_series_id')
        ]

        self.chunk_extractors = [
            ChunkExtractor(df, self.chunk_specs) for df in splitted_dfs
        ]

        self.lengths = [
            len(df) - self.chunk_extractors[0].chunk_length + 1
            for df in splitted_dfs
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

    def convert_item_to_df(self, item):
        tag_to_names_dict = {
            spec.tag: spec.names
            for spec in self.chunk_specs
        }
        output = {}
        for tag, values in item.items():
            data = {}
            names = tag_to_names_dict[tag]
            for name, series in zip(names, values.T):
                data[name] = series
            df = pd.DataFrame(data=data)
            df = self.column_transformer.inverse_transform(df)
            output[tag] = df

        return output

    def plot_chunks(self):
        plot_chunks(self.chunk_specs)