import logging
from typing import List

import torch

from lm_eval.models.huggingface import HFLM

log = logging.getLogger(__name__)


class NanoChatNoToolLM(HFLM):
    """
    HF wrapper that runs nanochat-style greedy generation without tool execution (batch=1).
    """

    def _generate_greedy(
        self,
        input_ids: List[int],
        max_new_tokens: int = 512,
    ) -> List[int]:
        tok = self.tokenizer
        device = self.device

        assistant_end = tok.convert_tokens_to_ids("<|assistant_end|>")
        bos = tok.bos_token_id

        current_tokens = list(input_ids)
        num_generated = 0

        while num_generated < max_new_tokens:
            ids_tensor = torch.tensor([current_tokens], device=device)
            with torch.no_grad():
                logits = self.model(ids_tensor).logits[:, -1, :]
            next_token = int(logits[0].argmax().item())

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[gen] step=%d next=%s decoded='%s'",
                    num_generated,
                    next_token,
                    tok.decode([next_token]).replace("\n", "\\n"),
                )

            current_tokens.append(next_token)

            if next_token == assistant_end or next_token == bos:
                break

            num_generated += 1

        return current_tokens

    def _model_generate(self, context, stop, **kwargs):
        # context: torch.LongTensor of shape (1, seq_len)
        assert (
            context.shape[0] == 1
        ), "hf-nanochat-no-tool supports batch_size=1 for generation"
        max_gen_toks = kwargs.get("max_gen_toks", 256)
        full = self._generate_greedy(
            context[0].tolist(), max_new_tokens=max_gen_toks
        )
        return torch.tensor([full], device=self.device)
