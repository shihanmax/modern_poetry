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
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def _get_initial_hidden(self, bs):
        h0 = torch.ones(2, bs, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, bs, self.hidden_dim).to(self.device)
        return h0, c0
    
    def forward_loss(self, src, tgt, valid_length):
        
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
        
        output = self.fc(output)
        output = output.permute(0, 2, 1)
        loss = self.cross_entropy_loss(output, tgt)
        
        return loss
    
    def forward(self, token_ids, max_decode_len):
        """Decode by given tokens.

        Args:
            token_ids (Tensor): bs, some_len
        """
        embeddings = self.embedding(token_ids)  # bs, some_len, embedding_dim
        
        result = []
        
        initial_hidden = self._get_initial_hidden(token_ids.shape[0])
        
        hidden = initial_hidden 
        for step in range(embeddings.shape[1]):
            curr_out, hidden = self.decode_one_step(
                embeddings[:, step, :].unsqueeze(1), hidden,
            )

            result.append(token_ids[:, step])  # bs, 
        
        inp = embeddings[:, -1, :].unsqueeze(1)

        while len(result) < max_decode_len:
            curr_out, hidden = self.decode_one_step(inp, hidden)
            
            # we'll do some sampling work here in the feature!
            token_idx = torch.argmax(curr_out[:, 0, :], dim=-1)
            token_idx = token_idx.squeeze()
            
            result.append(token_idx)

            inp = self.embedding(token_idx.unsqueeze(1))  # bs, 1, embed_dim
        
        result = torch.stack(result, dim=1)  # bs, max_decode_len 
        
        return result

    def decode_one_step(self, token_embedding, hidden):
        """Forward one step for decoding.

        Args:
            token_embedding (Tensor): bs, 1, embedding_dim
            
        Return:
            output (Tensor): bs, 1, hidden_dim
        """
        output, hidden = self.lstm(token_embedding, hidden)
        return output, hidden
