import numpy as np
import random

import logging as logging_master

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

logging_master.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging_master.WARN)
logging = logging_master.getLogger('corpus_stats')
logging.setLevel(logging_master.INFO)


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
        logging.info("Size of training file : {}".format(len(self.index)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        with open(self.training_file, 'r') as f:
            f.seek(self.index[index])
            line = f.readline().strip()
            try:
                parts = line.split('\t')
                lang1, lang1_str = parts[0].split('@')
                lang2, lang2_str = parts[1].split('@')
            except ValueError:
                logging.warning("line split error - {}".format(index))
                new_index = random.randint(0, len(self.index))
                return self.__getitem__(new_index)

            lang1_vec = np.fromstring(lang1_str, sep=',')
            lang2_vec = np.fromstring(lang2_str, sep=',')
            lang1_ten = torch.from_numpy(lang1_vec)
            lang2_ten = torch.from_numpy(lang2_vec)
            lang1_ten, lang2_ten = lang1_ten.type(torch.LongTensor), lang2_ten.type(torch.LongTensor)

            lang1_ten = Variable(lang1_ten)
            lang2_ten = Variable(lang2_ten)


        return lang1, lang1_ten, lang2, lang2_ten

    def __len__(self):
        return len(self.index)


