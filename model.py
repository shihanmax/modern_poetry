import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Generator(nn.Module):
    
    def __init__(
        self, embeddings, num_embeddings, embedding_dim, hidden_dim, device
    ):
        super(Generator, self).__init__()
        self.device = device
        
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        
        self.embedding = self.embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=0,
        )
        
        self.lstm = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_embeddings,
            bias=False,
        )
        
        self.relu = nn.ReLU()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def _get_initial_hidden(self, bs):
        h0 = torch.zeros(2, bs, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, bs, self.hidden_dim).to(self.device)
        return h0, c0
    
    def forward_rnn(self, src, valid_length):
        _, max_src_len = src.shape[:2]
        
        embedding = self.embedding(src)  # bs, max_src_len, embedding_dim
        
        packed_repr = pack_padded_sequence(
            embedding, valid_length.cpu().to(torch.int64), batch_first=True,
            enforce_sorted=False,
        )

        # output: [bs, max_src_len, hidden * 2]
        # hn, cn: [2, bs, hidden]
        output, hidden = self.lstm(packed_repr)

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=max_src_len,
        )
        
        output = self.fc(output)
        
        return output, hidden

    def forward_loss(self, src, tgt, valid_length):
        # bs, max_len, vocab_size
        output, _ = self.forward_rnn(src, valid_length)
        output = output.transpose(-1, -2)
        
        loss = self.cross_entropy_loss(output, tgt)
        
        return loss
    
    def forward_decoding(self, hint_token_ids, valid_length, max_decode_len):
        """Decode by given tokens.

        Args:
            hint_token_ids (Tensor): bs, some_len
        """
        bs, hint_len = hint_token_ids.shape
        
        result = []

        _, hidden = self.forward_rnn(hint_token_ids, valid_length)
        
        inp = hint_token_ids[:, -1].unsqueeze(1)

        while len(result) < max_decode_len - hint_len:
            inp = self.embedding(inp)
            curr_out, hidden = self.lstm(inp, hidden)
            curr_out = self.fc(curr_out)
            
            # we'll do some sampling work here in the feature!
            inp_idx = torch.argmax(curr_out[:, 0, :], dim=-1)
            inp_idx = inp_idx.squeeze()
            result.append(inp_idx.view(bs))

            inp = inp_idx.view(bs, 1)  # bs, 1
        
        result = torch.stack(result, dim=1)  # bs, max_decode_len - hint_len
        result = torch.cat([hint_token_ids, result], dim=1)

        return result
