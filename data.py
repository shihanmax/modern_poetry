import torch


class Dataset(object):
    
    def __init__(
        self, str2idx, sos_idx, eos_idx, pad_idx, unk_idx, max_seq_len, 
        all_poems,
    ):
        self.all_poems = all_poems
        self.str2idx = str2idx
        self.sos = sos_idx
        self.eos = eos_idx
        self.pad = pad_idx
        self.unk = unk_idx
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.all_poems)

    def __getitem__(self, item):
        poem = self.all_poems[item]
        return self.handle_one_poem(poem)
    
    def handle_one_poem(self, poem):
        body = poem["body"]
        src = [self.str2idx.get(token, self.unk) for token in body]
        src = [self.sos] + src[:self.max_seq_len - 1]  # truncate
        tgt = src[1:] + [self.eos] 
        
        valid_length = len(src)
        src.extend([self.pad] * (self.max_seq_len - valid_length))
        tgt.extend([self.pad] * (self.max_seq_len - valid_length))

        return {
            "src": torch.tensor(src),
            "tgt": torch.tensor(tgt),
            "valid_length": torch.tensor(valid_length)
        }
