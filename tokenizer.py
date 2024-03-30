from typing import List, Iterator

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext    

class Tokenizer:
    """Tokenizer class for tokenizing and detokenizing text data.
    """
    
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, src_model: str, tgt_model: str, dataset: torchtext.datasets) -> None:
        """Constructor for Tokenizer class.

        :param src_model: the name of the source language model
        :type src_model: str
        :param tgt_model: the name of the target language model
        :type tgt_model: str
        :param dataset: the dataset to build the vocabulary from
        :type dataset: torchtext.datasets
        :return: None
        """
        
        self.__src = get_tokenizer('spacy', language=src_model)
        self.__tgt = get_tokenizer('spacy', language=tgt_model)

        self.__vocab_src = build_vocab_from_iterator(self.__yield_tokens(dataset),
                                                     min_freq=1,
                                                     specials=self.special_symbols,
                                                     special_first=True)

        self.__vocab_tgt = build_vocab_from_iterator(self.__yield_tokens(dataset, src=False),
                                                     min_freq=1,
                                                     specials=self.special_symbols,
                                                     special_first=True)
        self.__vocab_src.set_default_index(0)
        self.__vocab_tgt.set_default_index(0)

    def __yield_tokens(self, data_iter: torchtext.datasets, src: bool = True) -> Iterator:
        """Function to yield tokens from a dataset.

        :param data_iter: the dataset to yield tokens from
        :type data_iter: torchtext.datasets
        :param src: the vocab model to use, defaults to True
        :type src: bool, optional
        :yield: the tokens from the dataset
        :rtype: Iterator
        """
        
        for data_sample in data_iter:
            if src:
                yield self.tokenize(data_sample[0], src)
            else:
                yield self.tokenize(data_sample[1], src)

    def tokenize(self, text: str, src: bool) -> str:
        """Function to tokenize text data.

        :param text: the text data to tokenize
        :type text: str
        :param src: the vocab model to use, defaults to True
        :type src: bool
        :return: the tokenized text data
        :rtype: str
        """
        
        if src:
            return self.__src(text)

        return self.__tgt(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Function to detokenize text data.

        :param tokens: the tokenized text data
        :type tokens: List[str]
        :return: the detokenized text data
        :rtype: str
        """
        
        return ' '.join(tokens[1:-1])

    def string_to_vocab(self, text: str, src: bool = True) -> torch.Tensor:
        """Function to convert text data to vocabulary.

        :param text: the text data to convert
        :type text: str
        :param src: the model to use to convert the text, defaults to True
        :type src: bool, optional
        :return: the vocabulary representation of the text data
        :rtype: torch.Tensor
        """
        
        if not src:
            tokens = self.tokenize(text, src)

            return torch.cat((
                torch.tensor([self.special_symbols.index('<bos>')]),
                torch.tensor(self.__vocab_tgt(tokens)),
                torch.tensor([self.special_symbols.index('<eos>')])
            ))

        tokens = self.tokenize(text, src)

        return torch.cat((
            torch.tensor([self.special_symbols.index('<bos>')]),
            torch.tensor(self.__vocab_src(tokens)),
            torch.tensor([self.special_symbols.index('<eos>')])
        ))

    def vocab_to_string(self, vocab: torch.Tensor, src: bool) -> str:
        """Function to convert vocabulary to text data.

        :param vocab: the vocabulary to convert
        :type vocab: torch.Tensor
        :param src: the model to use to convert the vocabulary, defaults to True
        :type src: bool
        :return: the text data representation of the vocabulary
        :rtype: str
        """
        
        if not src:
            return self.detokenize(
                self.__vocab_tgt.lookup_tokens(vocab.tolist())
            )

        return self.detokenize(
            self.__vocab_src.lookup_tokens(vocab.tolist())
        )
