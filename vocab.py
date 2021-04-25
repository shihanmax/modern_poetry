from nlkit.word2vec import train_w2v_from_line_file, load_wv_model


class Vocab(object):
    
    def __init__(self, w2v_model_path, binary=False):
        keep_tokens = ("<pad>", "<unk>", "<sos>", "<eos>")
        
        self.str2idx, self.idx2str, self.embedding = load_wv_model(
            w2v_model_path, binary, keep_tokens,
        )
        
        self.pad = 0
        self.unk = 1
        self.sos = 2
        self.eos = 3


if __name__ == "__main__":

    train_w2v_from_line_file(
        train_from="./all_for_vocab.txt",
        save_to="word_embedding.wv",
        tokenize=None,
        epochs=200,
        binary=False,
        size=256,
        min_count=5,
        window=6,
        sg=1,
        hs=0,
        negative=1,
        seed=12,
        compute_loss=True,
    )

    vocab = Vocab("./word_embedding.wv")
