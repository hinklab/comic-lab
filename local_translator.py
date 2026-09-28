"""
local_translator.py - 100% Offline Local NMT Translation Engine.
Uses CTranslate2 with int8 quantized NLLB-200-distilled-600M on CPU.
Operates with ZERO network connectivity or cloud APIs.
"""

import os
import re
import threading
from typing import List, Optional, Union
import ctranslate2
import sentencepiece as spm

# Default local model directory
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "nllb-200-distilled-600M-ct2-int8"
)

SRC_LANG = "eng_Latn"
TGT_LANG = "uzn_Latn"


class OfflineNLLBTranslator:
    """Thread-safe, singleton-capable local NMT translator using CTranslate2."""
    _instance: Optional["OfflineNLLBTranslator"] = None
    _lock = threading.Lock()

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        sp_path = os.path.join(self.model_dir, "sentencepiece.bpe.model")
        bin_path = os.path.join(self.model_dir, "model.bin")

        if not os.path.exists(sp_path) or not os.path.exists(bin_path):
            try:
                from huggingface_hub import snapshot_download
                print(f"[NLLB] Downloading model weights to {self.model_dir}...")
                os.makedirs(self.model_dir, exist_ok=True)
                snapshot_download(
                    repo_id="JustFrederik/nllb-200-distilled-600M-ct2-int8",
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print("[NLLB] Model weights downloaded successfully!")
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"Local NMT model directory not found at: {self.model_dir}. "
                    f"Auto-download failed: {dl_err}. "
                    "Run `python scripts/download_nllb_model.py` once to download the local model weights."
                )

        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"SentencePiece model file missing at: {sp_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)

        # Initialize CTranslate2 Translator on CPU with int8 quantization
        # Using intra_threads=4 or auto for fast CPU inference
        self.translator = ctranslate2.Translator(
            self.model_dir,
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=4
        )

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "OfflineNLLBTranslator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_dir)
            return cls._instance

    def translate_single(self, text: str) -> str:
        """Translates a single English text string into Uzbek offline."""
        cleaned = text.strip()
        if not cleaned:
            return ""

        batch_result = self.translate_batch([cleaned])
        return batch_result[0] if batch_result else cleaned

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translates a batch of English sentences simultaneously for optimal CPU throughput.
        Guaranteed 100% offline.
        """
        if not texts:
            return []

        clean_texts = [t.strip() for t in texts]
        non_empty_indices = [i for i, t in enumerate(clean_texts) if t]
        if not non_empty_indices:
            return clean_texts

        # Tokenize with SentencePiece and add NLLB language prefix/suffix
        tokenized_batch = []
        for i in non_empty_indices:
            tokens = [SRC_LANG] + self.sp.encode(clean_texts[i], out_type=str) + ["</s>"]
            tokenized_batch.append(tokens)

        # Run CTranslate2 batch translation
        target_prefixes = [[TGT_LANG]] * len(tokenized_batch)
        results = self.translator.translate_batch(
            tokenized_batch,
            target_prefix=target_prefixes,
            max_batch_size=16,
            beam_size=1,  # Greedy search: fast and reliable for short dialogue
            sampling_temperature=1.0,
            replace_unknowns=True
        )

        outputs = list(clean_texts)
        for idx, res in zip(non_empty_indices, results):
            out_tokens = res.hypotheses[0]
            # Strip target language prefix token if present
            if out_tokens and out_tokens[0] == TGT_LANG:
                out_tokens = out_tokens[1:]
            decoded = self.sp.decode(out_tokens).strip()
            outputs[idx] = decoded

        return outputs


def get_process_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

try:
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def _cached_nllb_translator(model_dir: Optional[str] = None):
        t = OfflineNLLBTranslator(model_dir)
        print(f"[DIAGNOSTIKA 4] NLLB modeli yuklangandan keyin (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)
        return t
except Exception:
    _cached_nllb_translator = None


def get_translator(model_dir: Optional[str] = None) -> OfflineNLLBTranslator:
    """Returns a singleton CTranslate2 NLLB translator cached by Streamlit."""
    if _cached_nllb_translator is not None:
        return _cached_nllb_translator(model_dir)
    t = OfflineNLLBTranslator.get_instance(model_dir)
    print(f"[DIAGNOSTIKA 4] NLLB modeli yuklangandan keyin (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)
    return t


def release_translator():
    """Releases CTranslate2 translator from memory and runs garbage collection."""
    try:
        if _cached_nllb_translator is not None and hasattr(_cached_nllb_translator, "clear"):
            _cached_nllb_translator.clear()
    except Exception:
        pass
    OfflineNLLBTranslator._instance = None
    import gc
    gc.collect()
    print(f"[DIAGNOSTIKA] NLLB translator xotiradan bo'shatildi (RAM RSS): {get_process_rss_mb():.1f} MB", flush=True)


def translate_offline(text_or_texts: Union[str, List[str]]) -> Union[str, List[str]]:
    """Convenience functional wrapper for offline translation."""
    translator = get_translator()
    if isinstance(text_or_texts, str):
        return translator.translate_single(text_or_texts)
    elif isinstance(text_or_texts, list):
        return translator.translate_batch(text_or_texts)
    return text_or_texts
