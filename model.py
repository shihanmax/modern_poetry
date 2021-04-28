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
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        
        self.fc = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=num_embeddings,
        )
        
        self.relu = nn.ReLU()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def _get_initial_hidden(self, bs):
        h0 = torch.zeros(2, bs, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, bs, self.hidden_dim).to(self.device)
        return h0, c0
    
    def forward_lstm(self, src, valid_length):
        _, max_src_len = src.shape[:2]
        
        embedding = self.embedding(src)  # bs, max_src_len, embedding_dim
        
        packed_repr = pack_padded_sequence(
            embedding, valid_length.cpu().to(torch.int64), batch_first=True,
            enforce_sorted=False,
        )

        # output: [bs, max_src_len, hidden * 2]
        # hn, cn: [2, bs, hidden]
        output, (hn, cn) = self.lstm(packed_repr)

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=max_src_len,
        )
        
        return output, (hn, cn)

    def forward_loss(self, src, tgt, valid_length):
        output, _ = self.forward_lstm(src, valid_length)
        output = self.fc(output)  # bs, max_len, vocab_size
        output = output.view(-1, output.shape[-1])

        loss = self.cross_entropy_loss(
            output.view(-1, output.shape[-1]), 
            tgt.view(-1)
        )
        
        return loss
    
    def forward_decoding(self, token_ids, valid_length, max_decode_len):
        """Decode by given tokens.

        Args:
            token_ids (Tensor): bs, some_len
        """
        result = []
        
        _, hidden = self.forward_lstm(token_ids, valid_length)
        
        inp = token_ids[:, -1].unsqueeze(1)  # TODO

        while len(result) < max_decode_len:
            inp = self.embedding(inp)
            curr_out, hidden = self.lstm(inp, hidden)
            curr_out = self.fc(curr_out)
            
            # we'll do some sampling work here in the feature!
            token_idx = torch.argmax(curr_out[:, 0, :], dim=-1)
            token_idx = token_idx.squeeze()
            
            result.append(token_idx)

            inp = token_idx.unsqueeze(1)  # bs, 1

        result = torch.stack(result, dim=1)  # bs, max_decode_len 

        return result
