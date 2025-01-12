import jittor as jt


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = jt.array(src, dtype=dtype).squeeze(1)
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype)
