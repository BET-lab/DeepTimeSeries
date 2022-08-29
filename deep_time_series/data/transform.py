import numpy as np
import pandas as pd


class CategoryMapper:
    def __init__(self):
        self.map = {}
        self.inverse_map = {}
        self.dtype = None

    def fit(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1

        x = x.reshape(-1)
        self.map = {v: k for k, v in enumerate(np.unique(x))}
        self.inverse_map = {v: k for k, v in self.map.items()}
        self.dtype = x.dtype

    def transform(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1

        x = x.reshape(-1)

        x = np.array([self.map[v] for v in x], dtype=np.int64)

        return x

    def inverse_transform(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1

        x = x.reshape(-1)

        x = np.array([self.inverse_map[v] for v in x], dtype=self.dtype)

        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


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
        for name in ['__time_index', '__time_series_id']:
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