import sentencepiece as spm


class Tokenizer:
    """Class to handle tokenization using SentencePiece model"""

    def __init__(self, model_file: str = "data/bpe.model"):
        self.__s = spm.SentencePieceProcessor(model_file=model_file)

    def encode(self, text: str) -> list:
        """Function to encode text into tokens using SentencePiece model

        :param text: the text to be encoded
        :type text: str
        :return: the encoded tokens
        :rtype: list
        """

        return self.__s.encode(
            text, out_type=int, enable_sampling=True, alpha=0.1, nbest_size=-1
        )

    def decode(self, tokens: list) -> str:
        """Function to decode tokens into text using SentencePiece model

        :param tokens: the tokens to be decoded
        :type tokens: list
        :return: the decoded text
        :rtype: str
        """

        return self.__s.decode(tokens, ignore_unk=True)

    def get_size(self) -> int:
        """Function to get the size of the SentencePiece model

        :return: the size of the SentencePiece model
        :rtype: int
        """

        return self.__s.get_piece_size()
