from typing import Optional

from torch import nn, Tensor


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        x = tgt

        if self.norm_first:
            sa_attn_output, sa_attn_output_weights = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + sa_attn_output
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_attn_output, sa_attn_output_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + sa_attn_output)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x, sa_attn_output_weights

    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        attn_output, attn_output_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True)

        attn_output = self.dropout1(attn_output)
        return attn_output, attn_output_weights


class TransformerDecoder(nn.TransformerDecoder):
    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, list[Tensor]]:
        output = tgt
        weights = []

        for mod in self.layers:
            output, layer_weights = mod(
                output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            weights.append(layer_weights)

        return output, weights
