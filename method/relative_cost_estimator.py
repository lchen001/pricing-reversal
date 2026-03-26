"""
Relative Cost Estimator: Embedding + KNN approach.

Given a query and a reference model, estimates the cost ratio vector r,
where r_i = cost(model_i) / cost(reference_model), using KNN over
embedded historical queries.
"""

import json
import os
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np


class RelativeCostEstimator:
    """Estimates relative cost ratios across models using embedding-based KNN.

    Workflow:
        1. build_index(): load historical data, compute per-query cost ratios,
           embed all queries, and build a KNN-ready index.
        2. estimate(query): embed the query, find K nearest neighbors,
           and return averaged cost ratio vector.
    """

    def __init__(
        self,
        reference_model: str,
        models: list[str] | None = None,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        api_key: str | None = None,
        k: int = 5,
        data_dir: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        Args:
            reference_model: Model name to use as the denominator in cost ratios.
            models: List of model names. If None, auto-detected from data.
            embedding_provider: 'openai' or 'gemini'.
            embedding_model: Override the default embedding model name.
            api_key: API key for the embedding provider. If None, uses env var.
            k: Number of nearest neighbors for estimation.
            data_dir: Path to consolidated data directory.
            cache_dir: Directory for caching embeddings. Defaults to data_dir/../.cache
        """
        self.reference_model = reference_model
        self.models = models
        self.embedding_provider = embedding_provider.lower()
        self.api_key = api_key
        self.k = k
        self.data_dir = data_dir

        # Set default embedding model per provider
        if embedding_model is None:
            if self.embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-small"
            elif self.embedding_provider == "gemini":
                self.embedding_model = "gemini-embedding-001"
            else:
                raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        else:
            self.embedding_model = embedding_model

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        elif data_dir is not None:
            self.cache_dir = Path(data_dir).parent / ".cache"
        else:
            self.cache_dir = None

        # Built by build_index()
        self._queries: list[str] = []           # query texts
        self._query_keys: list[tuple] = []      # (dataset, index) identifiers
        self._cost_ratios: np.ndarray | None = None  # shape (n_queries, n_models)
        self._embeddings: np.ndarray | None = None   # shape (n_queries, embed_dim)
        self._model_list: list[str] = []        # ordered model list for the ratio vector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        data_dir: str | None = None,
        datasets: list[str] | None = None,
    ) -> "RelativeCostEstimator":
        """Load data, compute cost ratios, embed queries, build KNN index.

        Args:
            data_dir: Override data directory.
            datasets: List of dataset file prefixes to include (e.g. ['aime-hybrid']).
                      If None, use all datasets found.

        Returns:
            self (for chaining).
        """
        data_dir = data_dir or self.data_dir
        if data_dir is None:
            raise ValueError("data_dir must be provided either in __init__ or build_index.")

        self.data_dir = data_dir
        if self.cache_dir is None:
            self.cache_dir = Path(data_dir).parent / ".cache"

        # Step 1: Load data → per-query costs across models
        query_data = self._load_data(data_dir, datasets)

        # Step 2: Compute cost ratios
        self._compute_cost_ratios(query_data)

        # Step 3: Embed all queries
        self._embeddings = self._embed_queries_batched(self._queries)

        return self

    def estimate(self, query: str) -> dict[str, float]:
        """Estimate cost ratios for a new query via KNN.

        Args:
            query: The input query string.

        Returns:
            Dict mapping model_name → estimated cost ratio (relative to reference).
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self._embed_queries_batched([query])  # (1, dim)
        similarities = self._cosine_similarity(q_emb, self._embeddings)[0]  # (n,)

        # Top-K nearest neighbors
        k = min(self.k, len(similarities))
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        # Weighted average by similarity (softmax-like weighting)
        top_k_sims = similarities[top_k_idx]
        weights = top_k_sims / top_k_sims.sum()

        top_k_ratios = self._cost_ratios[top_k_idx]  # (k, n_models)
        estimated = np.average(top_k_ratios, axis=0, weights=weights)

        return {model: float(estimated[i]) for i, model in enumerate(self._model_list)}

    def estimate_batch(self, queries: list[str]) -> list[dict[str, float]]:
        """Estimate cost ratios for multiple queries.

        Args:
            queries: List of query strings.

        Returns:
            List of dicts, each mapping model_name → estimated cost ratio.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_embs = self._embed_queries_batched(queries)  # (n_q, dim)
        sim_matrix = self._cosine_similarity(q_embs, self._embeddings)  # (n_q, n_index)

        results = []
        k = min(self.k, sim_matrix.shape[1])
        for i in range(len(queries)):
            sims = sim_matrix[i]
            top_k_idx = np.argsort(sims)[-k:][::-1]
            top_k_sims = sims[top_k_idx]
            weights = top_k_sims / top_k_sims.sum()
            top_k_ratios = self._cost_ratios[top_k_idx]
            estimated = np.average(top_k_ratios, axis=0, weights=weights)
            results.append({m: float(estimated[j]) for j, m in enumerate(self._model_list)})
        return results

    @property
    def model_list(self) -> list[str]:
        """Ordered list of models in the cost ratio vector."""
        return list(self._model_list)

    @property
    def n_queries(self) -> int:
        """Number of indexed queries."""
        return len(self._queries)

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_data(
        self, data_dir: str, datasets: list[str] | None
    ) -> dict[tuple, dict[str, dict]]:
        """Load consolidated JSON files and group records by (dataset, index).

        Returns:
            query_data: { (dataset_prefix, record_index): {
                'query': str,
                'costs': { model_name: float }
            }}
        """
        data_path = Path(data_dir)
        json_files = sorted(data_path.glob("*.json"))

        # Parse filenames to extract (dataset_prefix, model_name)
        records_by_file: list[tuple[str, str, list]] = []
        for fpath in json_files:
            fname = fpath.stem  # e.g. "aime-hybrid-gpt-5.2"
            parsed = self._parse_filename(fname)
            if parsed is None:
                continue
            dataset_prefix, model_name = parsed

            if datasets is not None and dataset_prefix not in datasets:
                continue

            with open(fpath, "r") as f:
                data = json.load(f)

            records_by_file.append((dataset_prefix, model_name, data.get("records", [])))

        # Auto-detect models if not specified
        all_models = sorted(set(m for _, m, _ in records_by_file))
        if self.models is not None:
            self._model_list = list(self.models)
        else:
            self._model_list = all_models

        if self.reference_model not in self._model_list:
            raise ValueError(
                f"Reference model '{self.reference_model}' not found in model list: {self._model_list}"
            )

        # Group by (dataset_prefix, record_index) → costs per model
        query_data: dict[tuple, dict] = {}
        for dataset_prefix, model_name, records in records_by_file:
            if model_name not in self._model_list:
                continue
            for rec in records:
                idx = rec["index"]
                key = (dataset_prefix, idx)
                if key not in query_data:
                    query_data[key] = {
                        "query": rec.get("origin_query") or rec.get("prompt", ""),
                        "costs": {},
                    }
                query_data[key]["costs"][model_name] = rec.get("cost", 0.0)

        return query_data

    def _parse_filename(self, stem: str) -> tuple[str, str] | None:
        """Parse a filename stem like 'aime-hybrid-gpt-5.2' into (dataset_prefix, model_name).

        Strategy: try matching known models from the end of the stem.
        """
        # Collect all known model names (from self.models or common ones)
        known_models = set()
        if self.models:
            known_models.update(self.models)
        # Also try to detect model name by looking for common model name patterns
        # We'll try progressively shorter suffixes
        # First try with known models
        for model in sorted(known_models, key=len, reverse=True):
            suffix = f"-{model}"
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                return (prefix, model)

        # Fallback: try splitting at each hyphen from the right
        # Common model names: gpt-5.2-high, gpt-5.2, gpt-5-mini, gemini-3.1-pro-preview, etc.
        # We need a heuristic or the experiment config
        # Load known model names from config if available
        config_path = Path(self.data_dir).parent / "constant" / "experiment_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            config_models = [m["model_name"] for m in config.get("models", [])]
            # Also check for extra models in data (gpt-5.2, claude-opus-4.6)
            for model in sorted(config_models, key=len, reverse=True):
                suffix = f"-{model}"
                if stem.endswith(suffix):
                    prefix = stem[: -len(suffix)]
                    return (prefix, model)

        # Last resort: scan all JSON files to build model set, or use broad patterns
        # Try common model name patterns (longest match first)
        candidate_models = [
            "claude-opus-4.6-thinking",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "claude-haiku-4.5",
            "claude-opus-4.6",
            "gpt-5.2-high",
            "MiniMax-M2.5",
            "gpt-5-mini",
            "kimi-k2.5",
            "gpt-5.2",
        ]
        for model in candidate_models:
            suffix = f"-{model}"
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                return (prefix, model)

        return None

    # ------------------------------------------------------------------
    # Cost Ratio Computation
    # ------------------------------------------------------------------

    def _compute_cost_ratios(self, query_data: dict[tuple, dict]) -> None:
        """Compute cost ratios relative to reference model.

        Populates self._queries, self._query_keys, self._cost_ratios.
        Only includes queries where the reference model has a positive cost.
        """
        ref_idx = self._model_list.index(self.reference_model)
        queries = []
        keys = []
        ratios = []

        for key in sorted(query_data.keys()):
            entry = query_data[key]
            costs = entry["costs"]

            # Must have all models present with positive cost
            if not all(m in costs and costs[m] > 0 for m in self._model_list):
                continue

            ref_cost = costs[self.reference_model]

            row = [costs[m] / ref_cost for m in self._model_list]
            queries.append(entry["query"])
            keys.append(key)
            ratios.append(row)

        self._queries = queries
        self._query_keys = keys
        self._cost_ratios = np.array(ratios, dtype=np.float64)  # (n, n_models)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_queries_batched(self, queries: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed queries with caching, processing in batches.

        Returns:
            np.ndarray of shape (len(queries), embed_dim).
        """
        from tqdm import tqdm

        if batch_size is None:
            batch_size = 100 if self.embedding_provider == "gemini" else 128

        cache = self._load_cache()

        # Identify which queries need embedding
        to_embed_indices = []
        to_embed_texts = []
        for i, q in enumerate(queries):
            qhash = self._query_hash(q)
            if qhash not in cache:
                to_embed_indices.append(i)
                to_embed_texts.append(q)

        # Embed missing queries in batches (save cache after each batch)
        if to_embed_texts:
            n_batches = (len(to_embed_texts) + batch_size - 1) // batch_size
            for start in tqdm(range(0, len(to_embed_texts), batch_size),
                              total=n_batches, desc="Embedding queries"):
                batch = to_embed_texts[start : start + batch_size]
                batch_embs = self._call_embedding_api(batch)
                for text, emb in zip(batch, batch_embs):
                    cache[self._query_hash(text)] = emb
                self._save_cache(cache)

        # Assemble result
        result = []
        for q in queries:
            result.append(cache[self._query_hash(q)])
        return np.array(result, dtype=np.float64)

    def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """Call the embedding API for a batch of texts."""
        if self.embedding_provider == "openai":
            return self._embed_openai(texts)
        elif self.embedding_provider == "gemini":
            return self._embed_gemini(texts)
        else:
            raise ValueError(f"Unknown provider: {self.embedding_provider}")

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)  # falls back to OPENAI_API_KEY env var if None
        response = client.embeddings.create(input=texts, model=self.embedding_model)
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Google Gemini API."""
        from google import genai

        client = genai.Client(api_key=self.api_key)  # falls back to GOOGLE_API_KEY env var if None
        result = client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between rows of A and rows of B.

        Args:
            A: shape (m, d)
            B: shape (n, d)

        Returns:
            Similarity matrix of shape (m, n).
        """
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A_norm @ B_norm.T

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        """Get cache file path based on embedding provider and model."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.embedding_model.replace("/", "_")
        return self.cache_dir / f"embeddings_{self.embedding_provider}_{safe_name}.pkl"

    def _load_cache(self) -> dict[str, list[float]]:
        """Load embedding cache from disk."""
        path = self._cache_path()
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        """Save embedding cache to disk."""
        path = self._cache_path()
        with open(path, "wb") as f:
            pickle.dump(cache, f)

    @staticmethod
    def _query_hash(text: str) -> str:
        """Deterministic hash for a query string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Evaluation Helpers
    # ------------------------------------------------------------------

    def train_test_split(
        self, test_ratio: float = 0.2, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split indexed queries into train/test sets by random indices.

        Args:
            test_ratio: Fraction of data for test set.
            seed: Random seed for reproducibility.

        Returns:
            (train_indices, test_indices) as numpy arrays.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        n = len(self._queries)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        return perm[n_test:], perm[:n_test]

    def evaluate(
        self,
        test_ratio: float = 0.2,
        seed: int = 42,
        metric: str = "mape",
    ) -> dict[str, float]:
        """Evaluate with a train-test split.

        Uses test_ratio of queries as test set, remaining as train set.
        For each test query, finds KNN in the train set and estimates
        cost ratios, then computes error against ground truth.

        Args:
            test_ratio: Fraction of queries used for testing.
            seed: Random seed.
            metric: 'mape' (mean absolute percentage error)
                    or 'mae' (mean absolute error).

        Returns:
            Dict mapping model_name → average error on the test set.
        """
        train_idx, test_idx = self.train_test_split(test_ratio, seed)

        train_embs = self._embeddings[train_idx]
        train_ratios = self._cost_ratios[train_idx]
        test_embs = self._embeddings[test_idx]
        test_ratios = self._cost_ratios[test_idx]

        sim_matrix = self._cosine_similarity(test_embs, train_embs)
        k = min(self.k, len(train_idx))

        errors = np.zeros((len(test_idx), len(self._model_list)), dtype=np.float64)
        for i in range(len(test_idx)):
            sims = sim_matrix[i]
            top_k_idx = np.argsort(sims)[-k:][::-1]
            top_k_sims = sims[top_k_idx]
            weights = top_k_sims / top_k_sims.sum()
            predicted = np.average(train_ratios[top_k_idx], axis=0, weights=weights)
            actual = test_ratios[i]

            if metric == "mape":
                errors[i] = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-2)
            elif metric == "mae":
                errors[i] = np.abs(predicted - actual)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        mean_errors = errors.mean(axis=0)
        return {m: float(mean_errors[j]) for j, m in enumerate(self._model_list)}

    def pricing_baseline(
        self,
        model_info: list[dict],
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> dict[str, float]:
        """Baseline: estimate cost ratios using API pricing only.

        For each model, the predicted ratio is
            (input_price_i + output_price_i) / (input_price_ref + output_price_ref).
        This prediction is query-independent.

        Args:
            model_info: List of model dicts with keys
                        'model_name', 'input_price_per_MTok', 'output_price_per_MTok'.
            test_ratio: Fraction of queries used for testing.
            seed: Random seed.

        Returns:
            Dict mapping model_name → MAE on the test set.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Build pricing lookup
        pricing = {}
        for m in model_info:
            pricing[m["model_name"]] = (
                m["input_price_per_MTok"] + m["output_price_per_MTok"]
            )

        ref_price = pricing[self.reference_model]
        predicted_ratios = np.array(
            [pricing[m] / ref_price for m in self._model_list], dtype=np.float64
        )

        _, test_idx = self.train_test_split(test_ratio, seed)
        test_ratios = self._cost_ratios[test_idx]  # (n_test, n_models)

        # MAE: each test query gets the same prediction
        errors = np.abs(test_ratios - predicted_ratios[None, :])  # broadcast
        mean_errors = errors.mean(axis=0)
        return {m: float(mean_errors[j]) for j, m in enumerate(self._model_list)}

    def __repr__(self) -> str:
        status = "built" if self._embeddings is not None else "not built"
        return (
            f"RelativeCostEstimator("
            f"ref='{self.reference_model}', "
            f"models={len(self._model_list)}, "
            f"provider='{self.embedding_provider}', "
            f"k={self.k}, "
            f"index={status}, "
            f"n_queries={len(self._queries)})"
        )
