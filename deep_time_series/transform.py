import copy
import numpy as np
import pandas as pd


def _merge_data_frames(
    data_frames: pd.DataFrame | list[pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(data_frames, pd.DataFrame):
        data_frames = [data_frames]

    return pd.concat(data_frames).reset_index(drop=True)


class ColumnTransformer:
    def __init__(
        self,
        transformer_dict=None,
        transformer_tuples=None,
    ):
        if transformer_dict is None and transformer_tuples is None:
            raise Exception(
                'One of transformer_dict or transformer_tuples has to be given.'
            )

        if (transformer_dict is not None) and (transformer_tuples is not None):
            raise Exception(
                'Both transformer_dict or transformer_tuples are set.'
            )

        if transformer_dict is not None:
            self.transformer_dict = transformer_dict

        if transformer_tuples is not None:
            self.transformer_dict = {}
            for transformer, names in transformer_tuples:
                for name in names:
                    self.transformer_dict[name] = copy.deepcopy(transformer)

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
        # Do not use list(valid_name_set) to preserve order of elements.
        return [name for name in names if name in valid_name_set]

    def fit(
        self,
        data_frames: pd.DataFrame | list[pd.DataFrame]
    ) -> None:
        df = _merge_data_frames(data_frames)

        for name in self._get_valid_names(df.columns):
            transformer = self.transformer_dict[name]

            self._apply_to_single_feature(
                df[name], transformer.fit
            )

    def transform(
        self,
        data_frames: pd.DataFrame | list[pd.DataFrame]
    ) -> pd.DataFrame | list[pd.DataFrame]:
        single_df = isinstance(data_frames, pd.DataFrame)

        if single_df:
            data_frames = [data_frames]

        dfs = []
        for df in data_frames:
            data = {}
            for name in self._get_valid_names(df.columns):
                transformer = self.transformer_dict[name]

                data[name] = self._apply_to_single_feature(
                    df[name], transformer.transform
                )

            # self._append_index_and_id(data, df)
            dfs.append(pd.DataFrame(data=data, index=df.index))

        if single_df:
            return dfs[0]
        else:
            return dfs

    def fit_transform(
        self,
        data_frames: pd.DataFrame | list[pd.DataFrame]
    ) -> pd.DataFrame | list[pd.DataFrame]:
        self.fit(data_frames)
        return self.transform(data_frames)

    def inverse_transform(
        self,
        data_frames: pd.DataFrame | list[pd.DataFrame]
    ) -> pd.DataFrame | list[pd.DataFrame]:
        single_df = isinstance(data_frames, pd.DataFrame)

        if single_df:
            data_frames = [data_frames]

        dfs = []
        for df in data_frames:
            data = {}
            for name in self._get_valid_names(df.columns):
                transformer = self.transformer_dict[name]

                data[name] = self._apply_to_single_feature(
                    df[name], transformer.inverse_transform
                )

            # self._append_index_and_id(data, df)
            dfs.append(pd.DataFrame(data=data, index=df.index))

        if single_df:
            return dfs[0]
        else:
            return dfs