from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence


class InputProcessor:
    """Tokenizer-backed text/token conversion for chat inputs and outputs."""

    def __init__(self, model_path: Path | None):
        self._model_path = model_path
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

    def _get_tokenizer(self):
        if self._model_path is None:
            return None
        if self._tokenizer is not None:
            return self._tokenizer
        with self._tokenizer_lock:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer  # type: ignore
                except Exception:
                    from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        return self._tokenizer

    def decode_tokens(self, token_ids: Sequence[int]) -> str | None:
        tok = self._get_tokenizer()
        if tok is None:
            return None
        return tok.decode(
            [int(t) for t in token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def encode_chat_messages(self, messages: Sequence[dict]) -> list[int]:
        tok = self._get_tokenizer()
        if tok is None:
            text = "\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
            return [int(b) for b in text.encode("utf-8")]
        text = tok.apply_chat_template(
            conversation=[{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages],
            add_generation_prompt=True,
            tokenize=False,
        )
        return [int(t) for t in tok.encode(text, add_special_tokens=False)]

