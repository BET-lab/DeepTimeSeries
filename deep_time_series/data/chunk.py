class RangeChunkSpec:
    def __init__(self, tag, names, range_, dtype):
        self.tag = tag
        self.names = names
        self.range_ = range_
        self.dtype = dtype


class ChunkSpec:
    def __init__(self, tag, names, dtype, shift=0):
        self.tag = tag
        self.names = names
        self.shift = shift
        self.dtype = dtype

    def to_range_chunk_spec(self, encoding_length, decoding_length):
        range_ = (
            encoding_length - self.shift,
            encoding_length + decoding_length - self.shift
        )

        return RangeChunkSpec(
            tag=self.tag, names=self.names,
            range_=range_, dtype=self.dtype,
        )


class EncodingChunkSpec(ChunkSpec):
    def __init__(self, tag, names, dtype, shift=0):
        super().__init__(
            tag=f'encoding.{tag}',
            names=names,
            shift=shift,
            dtype=dtype,
        )


class DecodingChunkSpec(ChunkSpec):
    def __init__(self, tag, names, dtype, shift=0):
        super().__init__(
            tag=f'decoding.{tag}',
            names=names,
            shift=shift,
            dtype=dtype,
        )


class LabelChunkSpec(ChunkSpec):
    def __init__(self, tag, names, dtype, shift=0):
        super().__init__(
            tag=f'label.{tag}',
            names=names,
            shift=shift,
            dtype=dtype,
        )


class ChunkExtractor:
    def __init__(self, df, range_chunk_specs):
        # Check tag duplication.
        tags = [spec.tag for spec in range_chunk_specs]
        assert len(tags) == len(set(tags))

        self.range_chunk_specs = range_chunk_specs
        self.chunk_length = max(spec.range_[1] for spec in range_chunk_specs)

        self._preprocess(df)

    def _preprocess(self, df):
        self.data = {}
        for spec in self.range_chunk_specs:
            self.data[spec.tag] = df[spec.names].astype(spec.dtype).values

    def extract(self, start_time_index):
        chunk_dict = {}
        for spec in self.range_chunk_specs:
            array = self.data[spec.tag][
                start_time_index : start_time_index+self.chunk_length
            ]

            chunk_dict[spec.tag] = array[slice(*spec.range_)]

        return chunk_dict