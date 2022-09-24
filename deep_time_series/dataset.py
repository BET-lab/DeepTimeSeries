from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from deep_time_series.transform import ColumnTransformer

from .chunk import BaseChunkSpec, ChunkExtractor
from .plotting import plot_chunks


class Rescaler:
    def __init__(
        self,
        chunk_specs: BaseChunkSpec,
        column_transformer: ColumnTransformer,
    ):
        self.tag_to_names = {
            chunk_spec.tag.split('.')[1]: chunk_spec.names
            for chunk_spec in chunk_specs
        }
        self.column_transformer = column_transformer

    def __call__(
        self,
        data: dict[str, torch.Tensor],
        sample_index_name='sample_index',
        time_index_name='time_index',
        neglect_suffix=False,
    ):
        outputs = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()

            tokens = k.split('.')
            # Neglect items with suffix.
            if len(tokens) != 2 and (not neglect_suffix):
                continue

            # Extract core tag.
            tag = tokens[1]

            # Neglect unexpected tags.
            if tag not in self.tag_to_names:
                continue

            names = self.tag_to_names[tag]

            dfs = []
            for i, y in enumerate(v):
                df = pd.DataFrame(data={
                    name: values for name, values in zip(names, y.T)
                })
                df = self.column_transformer.inverse_transform(df)

                df[sample_index_name] = i
                df[time_index_name] = np.arange(len(df))
                dfs.append(df)

            df = pd.concat(dfs)
            df = df.set_index([sample_index_name, time_index_name])

            outputs[k] = df

        return outputs

    def rescale_batches(
        self,
        data_list: list[dict[str, torch.Tensor]],
        batch_index_name='batch_index',
        sample_index_name='sample_index',
        time_index_name='time_index',
        neglect_suffix=False,
    ):
        batch_outputs = defaultdict(list)
        for i, data in enumerate(data_list):
            outputs = self(
                data=data,
                sample_index_name=sample_index_name,
                time_index_name=time_index_name,
                neglect_suffix=neglect_suffix
            )

            for k, v in outputs.items():
                v['batch_index'] = i
                batch_outputs[k].append(v)

        for k, v in batch_outputs.items():
            batch_outputs[k] = pd.concat(v).reset_index().set_index(
                [batch_index_name, sample_index_name, time_index_name]
            )

        return dict(batch_outputs)



class TimeSeriesDataset(Dataset):
    def __init__(self,
        data_frames: pd.DataFrame | list[pd.DataFrame],
        chunk_specs,
        # column_transformer,
        # fit_column_transformer=True,
        return_time_index=False,
    ):
        if isinstance(data_frames, pd.DataFrame):
            data_frames = [data_frames]
        self.data_frames = data_frames
        # self.data_frames = _merge_data_frames(data_frames=data_frames)
        # Make chunk_specs from encoding, decoding and label specs.
        self.chunk_specs = chunk_specs

        # self.column_transformer = column_transformer
        # self.fit_column_transformer = fit_column_transformer
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