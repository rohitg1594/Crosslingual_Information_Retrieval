import numpy as np
import random

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def list_line_locations(filename):
    line_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
    return line_offset


class EncodingDataset(object):

    def __init__(self, training_file):
        """
        data_path: directory path of training data files
        """
        super().__init__()

        self.training_file = training_file
        self.index = list_line_locations(self.training_file)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        if index in self.index:

            with open(self.training_file, 'r') as f:
                f.seek(self.index[index])
                line = f.readline().strip()
                try:
                    target, source = line.split('\t')
                except ValueError:
                    # print('TAB ERROR- {} {} {} {} {}'.format(index,index, line, shard_no, line_no))
                    new_index = random.randint(0, len(self.index))
                    return self.__getitem__(new_index)

                target = np.fromstring(target, sep=',', dtype=np.int)
                source = np.fromstring(source, sep=',', dtype=np.int)

                target = Variable(torch.from_numpy(target))
                source = Variable(torch.from_numpy(source))

            return source, target

        else:
            # print('MASTER INDEX ERROR - {}'.format(index))
            new_index = random.randint(0, self.len_ent)
            return self.__getitem__(new_index)

    def __len__(self):
        return len(self.index)
