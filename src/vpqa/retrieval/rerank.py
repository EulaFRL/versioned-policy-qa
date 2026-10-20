from __future__ import annotations

from vpqa.schema import Passage


class Reranker:
    """Cross-encoder rerank of a candidate list (bge-reranker-v2-m3 by default)."""

    def __init__(self, passages: list[Passage], model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 512, batch_size: int = 32, fp16: bool = True):
        import torch
        from sentence_transformers import CrossEncoder

        # fp16 is ~3.5x faster on Apple MPS with scores unchanged to 3 significant digits
        dtype = torch.float16 if fp16 and torch.backends.mps.is_available() else torch.float32
        self.model = CrossEncoder(model_name, max_length=max_length, model_kwargs={"torch_dtype": dtype})
        self.text = {p.pid: p.text for p in passages}
        self.batch_size = batch_size

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, pid) pairs. Pairs are batched by length so padding stays small."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(self.text[pairs[i][1]]))
        scores = self.model.predict([(pairs[i][0], self.text[pairs[i][1]]) for i in order],
                                    batch_size=self.batch_size, show_progress_bar=len(pairs) > 1000)
        out = [0.0] * len(pairs)
        for i, s in zip(order, scores):
            out[i] = float(s)
        return out

    def rerank(self, query: str, pids: list[str], top_n: int | None = None) -> list[str]:
        scores = self.score_many([(query, p) for p in pids])
        ranked = [p for _, p in sorted(zip(scores, pids), key=lambda x: -x[0])]
        return ranked[:top_n] if top_n else ranked
