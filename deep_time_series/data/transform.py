import numpy as np
import pandas as pd


class FeatureTransformers:
    def __init__(self, transformer_dict):
        self.transformer_dict = transformer_dict

    def _apply_to_single_feature(self, series, func):
        values = series.values.reshape(-1, 1)
        return_value = func(values)
        if isinstance(return_value, np.ndarray):
            return return_value.reshape(-1)
        else:
            return return_value

    def _append_index_and_id(self, data, df):
        for name in ['time_index', 'time_series_id']:
            if name in df.columns:
                data[name] = df[name]

    def _get_valid_names(self, names):
        valid_name_set = set(self.transformer_dict.keys()) & set(names)
        return [
            name for name in names if name in valid_name_set
        ]

    def fit(self, df):
        for name in self._get_valid_names(df.columns):
            transformer = self.transformer_dict[name]

            self._apply_to_single_feature(
                df[name], transformer.fit
            )

    def transform(self, df):
        data = {}
        for name in self._get_valid_names(df.columns):
            transformer = self.transformer_dict[name]

            data[name] = self._apply_to_single_feature(
                df[name], transformer.transform
            )

        self._append_index_and_id(data, df)
        return pd.DataFrame(data=data, index=df.index)

    def fit_transform(self, df):
        data = {}
        for name in self._get_valid_names(df.columns):
            transformer = self.transformer_dict[name]

            data[name] = self._apply_to_single_feature(
                df[name], transformer.fit_transform
            )

        self._append_index_and_id(data, df)
        return pd.DataFrame(data=data, index=df.index)

    def inverse_transform(self, df):
        data = {}
        for name in self._get_valid_names(df.columns):
            transformer = self.transformer_dict[name]

            data[name] = self._apply_to_single_feature(
                df[name], transformer.inverse_transform
            )

        self._append_index_and_id(data, df)
        return pd.DataFrame(data=data, index=df.index)
