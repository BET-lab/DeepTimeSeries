import numpy as np


class RangeChunkSpec:
    def __init__(self, tag, names, range_, dtype):
        self.tag = tag
        self.names = names
        self.range_ = range_
        self.dtype = dtype


class EncodingChunkSpec:
    def __init__(self, tag, names, dtype, shift=0):
        self.tag = tag
        self.names = names
        self.shift = shift
        self.dtype = dtype

    def to_range_chunk_spec(self, encoding_length, decoding_length):
        range_ = (
            self.shift,
            decoding_length + self.shift
        )

        return RangeChunkSpec(
            tag=f'encoding.{self.tag}', names=self.names,
            range_=range_, dtype=self.dtype,
        )


class DecodingChunkSpec:
    def __init__(self, tag, names, dtype, shift=0):
        self.tag = tag
        self.names = names
        self.shift = shift
        self.dtype = dtype

    def to_range_chunk_spec(self, encoding_length, decoding_length):
        range_ = (
            encoding_length + self.shift,
            encoding_length + decoding_length + self.shift
        )

        return RangeChunkSpec(
            tag=f'decoding.{self.tag}', names=self.names,
            range_=range_, dtype=self.dtype,
        )


class LabelChunkSpec:
    def __init__(self, tag, names, dtype, shift=0):
        self.tag = tag
        self.names = names
        self.shift = shift
        self.dtype = dtype

    def to_range_chunk_spec(self, encoding_length, decoding_length):
        range_ = (
            encoding_length + self.shift,
            encoding_length + decoding_length + self.shift
        )

        return RangeChunkSpec(
            tag=f'label.{self.tag}', names=self.names,
            range_=range_, dtype=self.dtype,
        )


class ChunkExtractor:
    def __init__(self, df, range_chunk_specs):
        # Check tag duplication.
        tags = [spec.tag for spec in range_chunk_specs]
        assert len(tags) == len(set(tags))

        self.range_chunk_specs = range_chunk_specs
        self.chunk_min_t = min(spec.range_[0] for spec in range_chunk_specs)
        self.chunk_max_t = max(spec.range_[1] for spec in range_chunk_specs)
        self.chunk_length = self.chunk_max_t - self.chunk_min_t

        self._preprocess(df)

    def _preprocess(self, df):
        self.data = {}
        for spec in self.range_chunk_specs:
            values = df[spec.names].astype(spec.dtype).values
            if len(values.shape) == 1:
                values = values[:, np.newaxis]
            self.data[spec.tag] = values

    def extract(self, start_time_index):
        assert start_time_index + self.chunk_min_t >= 0

        chunk_dict = {}
        for spec in self.range_chunk_specs:
            array = self.data[spec.tag][
                start_time_index + self.chunk_min_t :
                start_time_index + self.chunk_max_t
            ]

            range_ = (
                spec.range_[0] - self.chunk_min_t,
                spec.range_[1] - self.chunk_min_t,
            )

            chunk_dict[spec.tag] = array[slice(*range_)]

        return chunk_dict