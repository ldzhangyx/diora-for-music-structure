from collections import deque

import torch
import numpy as np

from tqdm import tqdm

from diora.data.reading import NLIReader, PlainTextReader, ConllReader
from diora.data.batch_iterator import BatchIterator
from diora.data.embeddings import EmbeddingsReader, UNK_TOKEN
from diora.data.preprocessing import indexify, build_text_vocab
from diora.data.preprocessing import synthesize_training_data
from diora.logging.configuration import get_logger
from diora.blocks.negative_sampler import NegativeSampler, calculate_freq_dist


class ConsolidateDatasets(object):
    """
    A class for consolidating many datasets.
    """

    def __init__(self, datasets):
        self.datasets = datasets

    def reindex(self, sentences, inverse_mapping):
        def fn(s):
            for idx in s:
                yield inverse_mapping[idx]
        def queue(lst):
            q = deque(lst)
            while len(q) > 0:
                yield q.popleft()
        return [list(fn(s)) for s in tqdm(queue(sentences), desc='reindex')]

    def remap_embeddings(self, datasets, inverse_mapping_lst, master_word2idx):
        size = datasets[0]['embeddings'].shape[1]
        embeddings = np.zeros((len(master_word2idx), size), dtype=np.float32)
        for dset, old2master in zip(datasets, inverse_mapping_lst):
            idx_from, idx_to = zip(*old2master.items())
            embeddings[np.asarray(idx_to)] = dset['embeddings'][np.asarray(idx_from)]
        return embeddings

    def consolidate_word2idx(self, word2idx_lst):
        master_word2idx = {}
        inverse_mapping_lst = []

        for w2i in word2idx_lst:
            old2master = {}
            for w, idx in w2i.items():
                if w not in master_word2idx:
                    master_word2idx[w] = len(master_word2idx)
                old2master[idx] = master_word2idx[w]
            inverse_mapping_lst.append(old2master)

        return master_word2idx, inverse_mapping_lst

    def run(self):
        word2idx_lst = [x['word2idx'] for x in self.datasets]
        master_word2idx, inverse_mapping_lst = self.consolidate_word2idx(word2idx_lst)
        embeddings = self.remap_embeddings(self.datasets, inverse_mapping_lst, master_word2idx)
        for dset, inverse_mapping in zip(self.datasets, inverse_mapping_lst):
            dset['sentences'] = self.reindex(dset['sentences'], inverse_mapping)
            dset['word2idx'] = master_word2idx
            dset['embeddings'] = embeddings


class ReaderManager(object):
    def __init__(self, reader):
        super(ReaderManager, self).__init__()
        self.reader = reader
        self.logger = get_logger()

    def run(self, options, text_path, embeddings_path):
        reader = self.reader
        logger = self.logger

        logger.info('Reading text: {}'.format(text_path))
        reader_result = reader.read(text_path)
        sentences = reader_result['sentences']
        extra = reader_result['extra']
        metadata = reader_result.get('metadata', {})
        logger.info('len(sentences)={}'.format(len(sentences)))

        word2idx = build_text_vocab(sentences)
        logger.info('len(vocab)={}'.format(len(word2idx)))

        if 'embeddings' in metadata:
            logger.info('Using embeddings from metadata.')
            embeddings = metadata['embeddings']
            del metadata['embeddings']
        else:
            logger.info('Reading embeddings.')
            embeddings, word2idx = EmbeddingsReader().get_embeddings(
                options, embeddings_path, word2idx)

        unk_index = word2idx.get(UNK_TOKEN, None)
        logger.info('Converting tokens to indexes (unk_index={}).'.format(unk_index))
        sentences = indexify(sentences, word2idx, unk_index)

        return {
            "sentences": sentences,
            "embeddings": embeddings,
            "word2idx": word2idx,
            "extra": extra,
            "metadata": metadata,
        }


class ReconstructDataset(object):

    def initialize(self, options, text_path=None, embeddings_path=None, filter_length=0, data_type=None):

        dataset = np.load('/gpfsnyu/scratch/yz6492/ec2vae/data/data.npy', allow_pickle=True)
        melody, chord = dataset[0], dataset[1]

        melody_list = list()
        chord_list = list()
        for i, j in zip(melody, chord):
            melody_list.append(i[:min(i.shape[0], j.shape[0])])
            chord_list.append(j[:min(i.shape[0], j.shape[0])])

        length = np.concatenate(melody_list).shape[0]// (options.sequence_length * options.batch_size) * (options.sequence_length * options.batch_size)
        melody = np.concatenate(melody_list)[:length].reshape(-1, options.sequence_length, 130)
        chord = np.concatenate(chord_list)[:length].reshape(-1, options.sequence_length, 12)
        data = np.concatenate([melody, chord], axis = -1)
        np.random.shuffle(data)
        length = data.shape[0]

        # for adjusting NLP samples, word==embeddings, word2id, id2embeddings
        embeddings = list()
        word2idx = dict()
        counter = 0
        for line in data:
            for element in line:
                if tuple(element) not in word2idx:
                    word2idx[tuple(element)] = counter
                    embeddings.append(element)
                    counter += 1
        embeddings = np.array(embeddings)

        # now data should be a list
        data_list = list()
        for line in data:
            data_line = [word2idx[tuple(i)] for i in line]
            data_list.append(data_line)

        train_result = {'sentences': data_list[:int(0.9*length)], 'embeddings': embeddings, 'word2idx': word2idx, 'extra': None}
        valid_result = {'sentences': data_list[int(0.9*length):], 'embeddings': embeddings, 'word2idx': word2idx, 'extra': None}

        # result是一个dict，元素包含{sentences, embeddings, word2idx, extra}， 其中sentences是idx list的list
        return train_result, valid_result


def make_batch_iterator(options, dset, shuffle=True, include_partial=False, filter_length=0,
                        batch_size=None, length_to_size=None):
    sentences = dset['sentences']
    word2idx = dset['word2idx']
    extra = dset['extra']
    # metadata = dset['metadata']

    cuda = options.cuda
    multigpu = options.multigpu
    ngpus = 1
    if cuda and multigpu:
        ngpus = torch.cuda.device_count()

    vocab_size = len(word2idx)
    
    negative_sampler = None
    if options.reconstruct_mode in ('margin', 'softmax'):
        freq_dist = calculate_freq_dist(sentences, vocab_size)
        negative_sampler = NegativeSampler(freq_dist=freq_dist, dist_power=options.freq_dist_power)
    vocab_lst = [w for w, _ in sorted(word2idx.items(), key=lambda x: x[1])]

    batch_iterator = BatchIterator(
        sentences, extra=extra, shuffle=shuffle, include_partial=include_partial,
        filter_length=filter_length, batch_size=batch_size, rank=options.local_rank,
        cuda=cuda, ngpus=ngpus, negative_sampler=negative_sampler,
        vocab=vocab_lst, k_neg=options.k_neg,
        options_path=options.elmo_options_path,
        weights_path=options.elmo_weights_path,
        length_to_size=length_to_size,
        )

    # DIRTY HACK: Makes it easier to print examples later. Should really wrap this within the class.
    batch_iterator.word2idx = word2idx

    return batch_iterator
