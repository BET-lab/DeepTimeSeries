import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .chunk import ChunkExtractor


class TimeSeriesDataset(Dataset):
    def __init__(self,
        df,
        encoding_length,
        decoding_length,
        chunk_specs,
        feature_transformers,
        fit_feature_transformers=True,
    ):
        self.df = df.copy()
        self.encoding_length = encoding_length,
        self.decoding_length = decoding_length,
        # Make chunk_specs from encoding, decoding and label specs.
        self.chunk_specs = [
            spec.to_range_chunk_spec(encoding_length, decoding_length)
            for spec in chunk_specs
        ]

        self.feature_transformers = feature_transformers
        self.fit_feature_transformers = fit_feature_transformers

        self._preprocess()

    def _preprocess(self):
        self.df.sort_values(by='time_index', inplace=True)
        if self.fit_feature_transformers:
            self.feature_transformers.fit(self.df)

        self.scaled_df = self.feature_transformers.transform(self.df)

        splitted_dfs = [
            df for _, df in self.scaled_df.groupby('time_series_id')
        ]

        self.chunk_extractors = [
            ChunkExtractor(df, self.chunk_specs) for df in splitted_dfs
        ]

        self.lengths = [
            len(df) - self.chunk_extractors[0].chunk_length
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

        chunk_dict = chunk_extractor.extract(start_time_index)

        return chunk_dict

    def convert_item_to_df(self, item):
        tag_to_names_dict = {
            spec.tag: spec.names
            for spec in self.chunk_extractors[0].range_chunk_specs
        }
        output = {}
        for tag, values in item.items():
            data = {}
            names = tag_to_names_dict[tag]
            for name, series in zip(names, values.T):
                data[name] = series
            df = pd.DataFrame(data=data)
            df = self.feature_transformers.inverse_transform(df)
            output[tag] = df

        return output
