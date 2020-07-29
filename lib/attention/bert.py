import torch.nn as nn

from .transformer import TransformerBlock


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim=2048, hidden_dim=512, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.input_dim = input_dim
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden_dim * 4
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, attn_heads, hidden_dim * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        if mask is None:
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.input_linear(x)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
