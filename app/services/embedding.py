import asyncio
import hashlib
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from app.core.config import settings

logger = logging.getLogger("techradar.embedding")


def calculate_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class BaseEmbedder:
    """Base embedder interface."""

    async def embed(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError

    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError


class DeterministicEmbedder(BaseEmbedder):
    """Deterministic feature-hashing n-gram embedder."""

    def __init__(self, dimensions: Optional[int] = None):
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS

    async def embed(self, text: str, **kwargs) -> List[float]:
        dim = kwargs.get("dim") or self.dimensions
        words = re.findall(r"\w+", text.lower())
        topic_vec = [0.0] * dim
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            topic_vec[h % dim] += 1.5
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            topic_vec[h % dim] += 2.0
        for w in words:
            if len(w) >= 4:
                for j in range(len(w) - 3):
                    h = int(hashlib.md5(w[j:j+4].encode("utf-8")).hexdigest(), 16)
                    topic_vec[h % dim] += 0.4

        norm = math.sqrt(sum(x * x for x in topic_vec)) or 1.0
        return [x / norm for x in topic_vec]

    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [await self.embed(t, **kwargs) for t in texts]


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI neural embedder (defaults to voyage-4-lite)."""

    _lock: Optional[asyncio.Lock] = None
    _last_request_timestamp: float = 0.0
    _min_request_interval: float = 0.0

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        target_dimensions: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 21.0,
        allow_fallback: bool = False,
    ):
        self.api_key = api_key if api_key is not None else settings.VOYAGE_API_KEY
        self.model = model or settings.VOYAGE_MODEL
        self.target_dimensions = target_dimensions or settings.EMBEDDING_DIMENSIONS
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.allow_fallback = allow_fallback
        self.endpoint = "https://api.voyageai.com/v1/embeddings"
        self._fallback = DeterministicEmbedder(dimensions=self.target_dimensions)

    def _normalize_vector_length(self, vec: List[float], target_dim: int) -> List[float]:
        """Align vector length to target_dim while preserving unit norm for cosine similarity."""
        current_len = len(vec)
        if current_len == target_dim:
            return vec
        if current_len > target_dim:
            sliced = vec[:target_dim]
            norm = math.sqrt(sum(x * x for x in sliced)) or 1.0
            return [x / norm for x in sliced]
        # Pad with zeros (dot product and cosine distance are invariant for unit vectors)
        padded = vec + [0.0] * (target_dim - current_len)
        norm = math.sqrt(sum(x * x for x in padded)) or 1.0
        return [x / norm for x in padded]

    async def embed(self, text: str, input_type: Optional[str] = None, **kwargs) -> List[float]:
        results = await self.embed_batch([text], input_type=input_type, **kwargs)
        return results[0]

    async def embed_batch(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        if not self.api_key:
            if self.allow_fallback:
                logger.warning("Voyage API key missing. Falling back to deterministic embedder.")
                return await self._fallback.embed_batch(texts)
            raise ValueError("VOYAGE_API_KEY is not configured in environment or settings.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if input_type in ("query", "document"):
            payload["input_type"] = input_type

        # Voyage models natively support 256, 512, 1024, 2048
        if self.target_dimensions in (256, 512, 1024, 2048):
            payload["output_dimension"] = self.target_dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        lock = self._get_lock()
        async with lock:
            if VoyageEmbedder._min_request_interval > 0.0:
                elapsed = time.monotonic() - VoyageEmbedder._last_request_timestamp
                if elapsed < VoyageEmbedder._min_request_interval:
                    wait_time = VoyageEmbedder._min_request_interval - elapsed
                    logger.info(f"Voyage AI queue: pacing request ({wait_time:.1f}s delay for 3 RPM)...")
                    await asyncio.sleep(wait_time)

            for attempt in range(self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(self.endpoint, headers=headers, json=payload)
                        if resp.status_code == 429:
                            VoyageEmbedder._min_request_interval = 21.0
                            if attempt < self.max_retries:
                                raw_retry = resp.headers.get("retry-after")
                                wait_seconds = float(raw_retry) if raw_retry and raw_retry.isdigit() else self.retry_delay
                                logger.warning(
                                    f"Voyage AI rate limit hit (3 RPM). Waiting {wait_seconds:.1f}s for quota refresh "
                                    f"(attempt {attempt + 1}/{self.max_retries})..."
                                )
                                await asyncio.sleep(wait_seconds)
                                continue
                            else:
                                resp.raise_for_status()

                        resp.raise_for_status()
                        VoyageEmbedder._last_request_timestamp = time.monotonic()
                        data = resp.json()
                        raw_embeddings = [item["embedding"] for item in data.get("data", [])]
                        return [
                            self._normalize_vector_length(vec, self.target_dimensions)
                            for vec in raw_embeddings
                        ]
                except (httpx.RequestError,) as e:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Voyage AI transient network/protocol error ({type(e).__name__}: {e}). "
                            f"Retrying in 5.0s (attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(5.0)
                        continue
                    if self.allow_fallback:
                        logger.warning(f"Voyage AI network error after retries: {e}. Falling back to deterministic.")
                        return await self._fallback.embed_batch(texts)
                    raise
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < self.max_retries:
                        raw_retry = e.response.headers.get("retry-after")
                        wait_seconds = float(raw_retry) if raw_retry and raw_retry.isdigit() else self.retry_delay
                        logger.warning(
                            f"Voyage AI rate limit hit (3 RPM). Waiting {wait_seconds:.1f}s for quota refresh "
                            f"(attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue

                    if self.allow_fallback:
                        logger.warning(f"Voyage AI API error ({e.response.status_code}): {e.response.text}. Falling back to deterministic.")
                        return await self._fallback.embed_batch(texts)
                    raise
                except Exception as e:
                    if self.allow_fallback:
                        logger.warning(f"Voyage AI API call failed: {e}. Falling back to deterministic embedder.")
                        return await self._fallback.embed_batch(texts)
                    raise

        raise RuntimeError(f"Voyage AI requests exhausted after {self.max_retries} retries due to rate limits.")


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini neural embedder (defaults to text-embedding-004)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        target_dimensions: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        allow_fallback: bool = False,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        raw_model = model or settings.GEMINI_MODEL
        clean = raw_model.replace("models/", "")
        # Google v1beta uses gemini-embedding-001; map legacy or user aliases
        if clean in ("text-embedding-004", "embedding-004"):
            clean = "gemini-embedding-001"
        self.clean_model = clean
        self.model = clean
        self.target_dimensions = target_dimensions or settings.EMBEDDING_DIMENSIONS
        self.timeout = timeout
        self.max_retries = max_retries
        self.allow_fallback = allow_fallback
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.clean_model}:batchEmbedContents"
        )
        self._fallback = DeterministicEmbedder(dimensions=self.target_dimensions)

    def _normalize_vector_length(self, vec: List[float], target_dim: int) -> List[float]:
        """Align vector length to target_dim while preserving unit norm for cosine similarity."""
        current_len = len(vec)
        if current_len == target_dim:
            return vec
        if current_len > target_dim:
            sliced = vec[:target_dim]
            norm = math.sqrt(sum(x * x for x in sliced)) or 1.0
            return [x / norm for x in sliced]
        # Pad with zeros (dot product and cosine distance are invariant for unit vectors)
        padded = vec + [0.0] * (target_dim - current_len)
        norm = math.sqrt(sum(x * x for x in padded)) or 1.0
        return [x / norm for x in padded]

    async def embed(self, text: str, input_type: Optional[str] = None, **kwargs) -> List[float]:
        results = await self.embed_batch([text], input_type=input_type, **kwargs)
        return results[0]

    async def embed_batch(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        if not self.api_key:
            if self.allow_fallback:
                logger.warning("Gemini API key missing. Falling back to deterministic embedder.")
                return await self._fallback.embed_batch(texts)
            raise ValueError("GEMINI_API_KEY is not configured in environment or settings.")

        task_type = None
        if input_type == "query":
            task_type = "RETRIEVAL_QUERY"
        elif input_type == "document":
            task_type = "RETRIEVAL_DOCUMENT"

        requests = []
        for text in texts:
            req_item: Dict[str, Any] = {
                "model": f"models/{self.clean_model}",
                "content": {
                    "parts": [{"text": text}]
                },
            }
            if task_type:
                req_item["taskType"] = task_type
            if self.target_dimensions in (256, 512, 768, 1024, 1536, 2048, 3072):
                req_item["outputDimensionality"] = self.target_dimensions
            requests.append(req_item)

        payload = {"requests": requests}
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.endpoint}?key={self.api_key}",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code == 429:
                        if attempt < self.max_retries:
                            wait_seconds = 2.0 * (2 ** attempt)
                            logger.warning(
                                f"Gemini API rate limit hit. Waiting {wait_seconds:.1f}s "
                                f"(attempt {attempt + 1}/{self.max_retries})..."
                            )
                            await asyncio.sleep(wait_seconds)
                            continue
                        else:
                            resp.raise_for_status()

                    resp.raise_for_status()
                    data = resp.json()
                    raw_embeddings = [item["values"] for item in data.get("embeddings", [])]
                    return [
                        self._normalize_vector_length(vec, self.target_dimensions)
                        for vec in raw_embeddings
                    ]
            except (httpx.RequestError,) as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Gemini API transient network error ({type(e).__name__}: {e}). "
                        f"Retrying in 2.0s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(2.0)
                    continue
                if self.allow_fallback:
                    logger.warning(f"Gemini API network error after retries: {e}. Falling back to deterministic.")
                    return await self._fallback.embed_batch(texts)
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and self.clean_model != "gemini-embedding-001":
                    logger.warning(
                        f"Gemini model '{self.clean_model}' returned 404. Auto-recovering with 'gemini-embedding-001'..."
                    )
                    self.clean_model = "gemini-embedding-001"
                    self.model = "gemini-embedding-001"
                    self.endpoint = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.clean_model}:batchEmbedContents"
                    )
                    for req in requests:
                        req["model"] = f"models/{self.clean_model}"
                    continue

                if e.response.status_code == 429 and attempt < self.max_retries:
                    wait_seconds = 2.0 * (2 ** attempt)
                    logger.warning(
                        f"Gemini API rate limit hit (429). Waiting {wait_seconds:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                if self.allow_fallback:
                    logger.warning(f"Gemini API error ({e.response.status_code}): {e.response.text}. Falling back to deterministic.")
                    return await self._fallback.embed_batch(texts)
                raise
            except Exception as e:
                if self.allow_fallback:
                    logger.warning(f"Gemini API call failed: {e}. Falling back to deterministic embedder.")
                    return await self._fallback.embed_batch(texts)
                raise

        raise RuntimeError(f"Gemini API requests exhausted after {self.max_retries} retries.")


def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """Get embedder instance by provider name ('deterministic', 'voyage', or 'gemini')."""
    selected = (provider or settings.EMBEDDING_PROVIDER).strip().lower()
    if selected in ("gemini", "google", "text-embedding-004", "gemini-embedding-001"):
        return GeminiEmbedder()
    if selected in ("voyage", "voyageai"):
        return VoyageEmbedder()
    return DeterministicEmbedder()


async def get_embedding(
    text: str,
    provider: Optional[str] = None,
    input_type: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Generate embedding for text using selected provider ('deterministic', 'voyage', or 'gemini')."""
    embedder = get_embedder(provider)
    if isinstance(embedder, (VoyageEmbedder, GeminiEmbedder)):
        return await embedder.embed(text, input_type=input_type, **kwargs)
    return await embedder.embed(text, **kwargs)


async def generate_embedding(
    text: str,
    provider: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Compatibility alias for get_embedding."""
    return await get_embedding(text, provider=provider, **kwargs)


async def compare_embeddings(text_a: str, text_b: str) -> Dict[str, Any]:
    """Compare similarity scores and output vectors between Deterministic, Voyage AI, and Gemini embedders."""
    det = DeterministicEmbedder()
    voyage = VoyageEmbedder()

    det_vec_a = await det.embed(text_a)
    det_vec_b = await det.embed(text_b)
    det_sim = calculate_cosine_similarity(det_vec_a, det_vec_b)

    voyage_vecs = await voyage.embed_batch([text_a, text_b])
    voyage_vec_a, voyage_vec_b = voyage_vecs[0], voyage_vecs[1]
    voyage_sim = calculate_cosine_similarity(voyage_vec_a, voyage_vec_b)

    output: Dict[str, Any] = {
        "text_a": text_a,
        "text_b": text_b,
        "deterministic": {
            "similarity": round(det_sim, 4),
            "dimensions": len(det_vec_a),
        },
        "voyage": {
            "similarity": round(voyage_sim, 4),
            "model": voyage.model,
            "dimensions": len(voyage_vec_a),
        },
        "absolute_difference": round(abs(det_sim - voyage_sim), 4),
    }

    if settings.GEMINI_API_KEY:
        try:
            gemini = GeminiEmbedder()
            gem_vecs = await gemini.embed_batch([text_a, text_b])
            gem_sim = calculate_cosine_similarity(gem_vecs[0], gem_vecs[1])
            output["gemini"] = {
                "similarity": round(gem_sim, 4),
                "model": gemini.model,
                "dimensions": len(gem_vecs[0]),
            }
        except Exception as ge:
            logger.warning(f"Could not evaluate Gemini embedding in compare_embeddings: {ge}")

    return output


async def compare_embeddings_batch(
    pairs: List[Tuple[str, str]],
    labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare multiple pairs across embedders in unified batch requests."""
    det = DeterministicEmbedder()
    voyage = VoyageEmbedder()

    all_texts: List[str] = []
    text_to_idx: Dict[str, int] = {}
    for text_a, text_b in pairs:
        if text_a not in text_to_idx:
            text_to_idx[text_a] = len(all_texts)
            all_texts.append(text_a)
        if text_b not in text_to_idx:
            text_to_idx[text_b] = len(all_texts)
            all_texts.append(text_b)

    det_vectors = await det.embed_batch(all_texts)
    voyage_vectors = await voyage.embed_batch(all_texts)

    gemini_vectors: Optional[List[List[float]]] = None
    gemini_model: str = settings.GEMINI_MODEL
    if settings.GEMINI_API_KEY:
        try:
            gemini = GeminiEmbedder()
            gemini_model = gemini.model
            gemini_vectors = await gemini.embed_batch(all_texts)
        except Exception as ge:
            logger.warning(f"Could not evaluate Gemini batch in compare_embeddings_batch: {ge}")

    results: List[Dict[str, Any]] = []
    for i, (text_a, text_b) in enumerate(pairs):
        idx_a = text_to_idx[text_a]
        idx_b = text_to_idx[text_b]

        det_sim = calculate_cosine_similarity(det_vectors[idx_a], det_vectors[idx_b])
        voyage_sim = calculate_cosine_similarity(voyage_vectors[idx_a], voyage_vectors[idx_b])

        item: Dict[str, Any] = {
            "text_a": text_a,
            "text_b": text_b,
            "deterministic": {
                "similarity": round(det_sim, 4),
                "dimensions": len(det_vectors[idx_a]),
            },
            "voyage": {
                "similarity": round(voyage_sim, 4),
                "model": voyage.model,
                "dimensions": len(voyage_vectors[idx_a]),
            },
            "absolute_difference": round(abs(det_sim - voyage_sim), 4),
        }
        if gemini_vectors:
            gem_sim = calculate_cosine_similarity(gemini_vectors[idx_a], gemini_vectors[idx_b])
            item["gemini"] = {
                "similarity": round(gem_sim, 4),
                "model": gemini_model,
                "dimensions": len(gemini_vectors[idx_a]),
            }
        if labels and i < len(labels):
            item["label"] = labels[i]
        results.append(item)

    return results
