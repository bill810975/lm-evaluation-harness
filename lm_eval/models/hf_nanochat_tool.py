import logging
from collections import deque
from typing import List, Optional

import torch
import transformers

from lm_eval.models.huggingface import HFLM

log = logging.getLogger(__name__)


# Helpers ported from nanochat (simplified timeout handling)
def _safe_eval(expr: str):
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def use_calculator(expr: str):
    """
    Evaluate a Python expression safely (subset), mirroring nanochat/engine.py use_calculator.
    Supports simple math and limited string .count.
    """
    expr = expr.replace(",", "")
    # pure math
    if all([x in "0123456789*+-/.() " for x in expr]) and "**" not in expr:
        return _safe_eval(expr)
    # string count guardrails
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None
    dangerous = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    if any(p in expr.lower() for p in dangerous):
        return None
    if ".count(" not in expr:
        return None
    return _safe_eval(expr)


class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class NanoChatToolLM(HFLM):
    """
    HF wrapper that emulates nanochat Engine.generate tool loop (greedy, batch=1).
    """

    def _generate_with_tools(
        self,
        input_ids: List[int],
        max_new_tokens: int = 512,
    ) -> List[int]:
        tok = self.tokenizer
        device = self.device

        get_id = tok.convert_tokens_to_ids
        py_start = get_id("<|python_start|>")
        py_end = get_id("<|python_end|>")
        out_start = get_id("<|output_start|>")
        out_end = get_id("<|output_end|>")
        assistant_end = get_id("<|assistant_end|>")
        bos = tok.bos_token_id

        state = RowState(list(input_ids))
        num_generated = 0

        step = 0
        while num_generated < max_new_tokens and not state.completed:
            # Choose next token (forced tokens first)
            if state.forced_tokens:
                next_token = state.forced_tokens.popleft()
            else:
                ids_tensor = torch.tensor([state.current_tokens], device=device)
                with torch.no_grad():
                    logits = self.model(ids_tensor).logits[:, -1, :]
                next_token = int(logits[0].argmax().item())

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[gen] step=%d forced=%s next=%s decoded='%s'",
                    num_generated,
                    state.forced_tokens.__len__() > 0,
                    next_token,
                    tok.decode([next_token]).replace("\n", "\\n"),
                )

            state.current_tokens.append(next_token)

            # Stop if assistant_end or bos
            if next_token == assistant_end or next_token == bos:
                state.completed = True
                break

            # Tool logic
            if next_token == py_start:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"[tool] step {step}: enter python block")
                state.in_python_block = True
                state.python_expr_tokens = []
            elif next_token == py_end and state.in_python_block:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"[tool] step {step}: exit python block")
                state.in_python_block = False
                if state.python_expr_tokens:
                    expr = tok.decode(state.python_expr_tokens)
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"[tool] expr='{expr}'")
                    result = use_calculator(expr)
                    if result is not None:
                        result_tokens = tok.encode(str(result), add_special_tokens=False)
                        state.forced_tokens.append(out_start)
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(out_end)
                        if log.isEnabledFor(logging.DEBUG):
                            log.debug(f"[tool] result={result}")
                state.python_expr_tokens = []
            elif state.in_python_block:
                state.python_expr_tokens.append(next_token)

            num_generated += 1
            step += 1

        return state.current_tokens

    def _model_generate(self, context, stop, **kwargs):
        # context: torch.LongTensor of shape (1, seq_len)
        assert (
            context.shape[0] == 1
        ), "hf-nanochat-tool supports batch_size=1 for generation"
        max_gen_toks = kwargs.get("max_gen_toks", 256)
        full = self._generate_with_tools(
            context[0].tolist(), max_new_tokens=max_gen_toks
        )
        return torch.tensor([full], device=self.device)
