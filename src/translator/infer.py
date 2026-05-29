"""vLLM-based translator inference with optional LoRA adapter.

This module owns the heavyweight LLM handle. Importing it triggers a vLLM
load (slow), so the pipeline keeps a single shared instance via
`get_translator()`. Tests can stub out `LLMBackend` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from data.types import AnswerType, Record, Translation
from translator.parse import parse_translator_output
from translator.prompt import build_messages_for_record


@dataclass
class TranslatorConfig:
    # Defaults target Qwen3.5-4B on an RTX 5070 (12 GB GDDR7, Blackwell).
    # Override `model` and the GPU knobs from configs/default.yaml for other setups.
    model: str = "Qwen/Qwen3.5-4B"
    quantization: str | None = "awq_marlin"   # awq_marlin | fp8 | None
    max_model_len: int = 4096                 # half the 4090 default to keep KV cache modest on 12 GB
    gpu_memory_utilization: float = 0.80      # leave ~2 GB free for Z3 + Python
    enable_lora: bool = True
    max_lora_rank: int = 32
    lora_path: str | None = None              # set after fine-tune
    k_samples: int = 5
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 1024
    n_fewshot: int = 2


class LLMBackend(Protocol):
    def chat_generate(
        self,
        batch_messages: list[list[dict]],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        lora_path: str | None,
    ) -> list[list[str]]:
        """Return per-prompt, per-sample raw completion strings."""
        ...


class VLLMBackend:
    def __init__(self, cfg: TranslatorConfig):
        # Imports are lazy so the rest of the package can be imported on a box
        # without vLLM installed (Windows dev, tests, etc.).
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=cfg.model,
            quantization=cfg.quantization,
            dtype="auto",
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_lora=cfg.enable_lora,
            max_lora_rank=cfg.max_lora_rank,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def chat_generate(
        self,
        batch_messages: list[list[dict]],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        lora_path: str | None,
    ) -> list[list[str]]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        sp = self._SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest  # type: ignore[import-not-found]

            lora_request = LoRARequest("translator", 1, lora_path)
        outputs = self.llm.generate(prompts, sampling_params=sp, lora_request=lora_request)
        return [[c.text for c in out.outputs] for out in outputs]


class Translator:
    def __init__(self, backend: LLMBackend, cfg: TranslatorConfig):
        self.backend = backend
        self.cfg = cfg

    def translate(self, record: Record) -> list[list[Translation]]:
        """Return per-prompt list of K Translation candidates.

        - Yes/No/Uncertain → outer list has 1 entry, inner list has K.
        - MCQ → outer list has len(options) entries, inner list has K each.
        - Open-ended → empty outer list.
        """
        if record.answer_type == AnswerType.OPEN_ENDED:
            return []

        batches = build_messages_for_record(record, n_fewshot=self.cfg.n_fewshot)
        if not batches:
            return []

        raw = self.backend.chat_generate(
            batch_messages=batches,
            n=self.cfg.k_samples,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
            lora_path=self.cfg.lora_path,
        )

        translations: list[list[Translation]] = []
        for prompt_outputs in raw:
            per_prompt: list[Translation] = []
            for i, txt in enumerate(prompt_outputs):
                code = parse_translator_output(txt)
                if code is None:
                    continue
                per_prompt.append(
                    Translation(code=code, raw_text=txt, sample_index=i)
                )
            translations.append(per_prompt)
        return translations


_GLOBAL_TRANSLATOR: Translator | None = None


def get_translator(cfg: TranslatorConfig | None = None) -> Translator:
    """Lazy singleton — one vLLM load per process."""
    global _GLOBAL_TRANSLATOR
    if _GLOBAL_TRANSLATOR is None:
        cfg = cfg or TranslatorConfig()
        backend = VLLMBackend(cfg)
        _GLOBAL_TRANSLATOR = Translator(backend, cfg)
    return _GLOBAL_TRANSLATOR
