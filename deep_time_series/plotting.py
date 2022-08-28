import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_chunks(range_chunk_specs, encoding_length, decoding_length):
    ys = [0, 0]
    widths = [encoding_length, decoding_length]
    lefts = [0, encoding_length]
    tags = ['encoding', 'decoding']

    for y, spec in enumerate(range_chunk_specs, start=1):
        ys.append(y)
        widths.append(spec.range[1]-spec.range[0])
        lefts.append(spec.range[0])
        tags.append(spec.tag)

    for y, width, left, tag in zip(ys, widths, lefts, tags):
        plt.barh(y=y, width=width, left=left, alpha=0.8)
        plt.annotate(tag, (left, y))