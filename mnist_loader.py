# Copyright 2020 Andrew Owen Martin
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import struct


def get_60k_data(max_items=None):

    data = get_py_data(max_items=max_items, path="./datasets/train-images-idx3-ubyte")
    labels = list(get_labels("./datasets/train-labels-idx1-ubyte", max_items=max_items))

    return data, labels


def get_10k_data(max_items=None):

    data = get_py_data(max_items=max_items, path="./datasets/t10k-images-idx3-ubyte")
    labels = list(get_labels("./datasets/t10k-labels-idx1-ubyte", max_items=max_items))

    return data, labels


def get_data(path):
    with open(path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        data = data.reshape((size, nrows, ncols))
        return data


def get_py_data(path, max_items):
    data = get_data(path)
    return data_to_lists(data, max_items=max_items)


def data_to_lists(data, max_items):
    data_list = []
    for char_num, char_bytes in enumerate(data):
        if char_num == max_items:
            break
        char_list = [list([int(col) for col in row]) for row in char_bytes]
        data_list.append(char_list)
    return data_list


def get_labels(path, max_items=None):
    with open(path, "rb") as f:
        a, b = struct.unpack(">II", f.read(8))
        count = 0
        while True:
            byte = f.read(1)
            if not byte:
                break
            num = int.from_bytes(byte, byteorder="little")
            yield num
            count += 1
            if max_items is not None and count >= max_items:
                break
