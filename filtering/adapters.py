"""Real-model adapters for CLIP-ITM, VQAScore, and BLIP Student scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class OpenClipAdapter:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        import open_clip
        import torch

        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()

    def embed_image(self, image):
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self.model.encode_image(tensor)
        return emb.cpu().numpy().squeeze(0)

    def embed_text(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)
        with self._torch.no_grad():
            emb = self.model.encode_text(tokens)
        return emb.cpu().numpy().squeeze(0)

    def embed_images_batch(self, images: list) -> list:
        tensors = []
        for image in images:
            if isinstance(image, (str, bytes)):
                image = Image.open(image).convert("RGB")
            tensors.append(self.preprocess(image))
        batch = self._torch.stack(tensors, dim=0).to(self.device)
        with self._torch.no_grad():
            emb = self.model.encode_image(batch)
        return [emb[i].cpu().numpy() for i in range(emb.size(0))]


class T2VVQAScoreBackend:
    """CLIP-FlanT5 VQAScore via t2v_metrics (lazy import)."""

    def __init__(
        self,
        model: str = "clip-flant5-xl",
        device: str = "cuda",
        **_: Any,
    ):
        import t2v_metrics

        self.model_name = model
        self.device = device
        self._scorer = t2v_metrics.VQAScore(model=model, device=device)

    def score_pairs(
        self, image_paths: list[str | Path], texts: list[str]
    ) -> list[float]:
        if len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have the same length")
        if not image_paths:
            return []
        paths = [str(p) for p in image_paths]
        # IMPORTANT: call model.forward directly for true paired scoring (n scores
        # for n pairs in one batched forward). The Score.__call__ wrapper instead
        # builds an m x n cross-product (m*n forwards), which makes throughput scale
        # quadratically with batch size — much slower for larger batches.
        model = getattr(self._scorer, "model", None)
        if model is not None and hasattr(model, "forward"):
            import torch

            with torch.no_grad():
                raw = model.forward(paths, texts)
            return _tensor_scores_to_list(raw, len(paths))
        raw = self._scorer(images=paths, texts=texts)
        return _tensor_scores_to_list(raw, len(paths))


class HuggingFaceYesNoVLMAdapter:
    """VQAScore-like scorer for Hugging Face VLMs with next-token logits.

    This backend turns a candidate image-text match into a yes/no VQA prompt and
    scores the next-token probability of yes against no. It is intentionally
    model-agnostic: PA-LLaVA, LLaVA-Med, or another VLM can be plugged in as long
    as it exposes a Transformers processor and conditional-generation model.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "USER: {image}\n"
        "Does this pathology image support the following question-answer "
        "statement?\n{text}\nAnswer yes or no.\nASSISTANT:"
    )
    DEFAULT_YES_TOKENS = ("Yes", " yes", "yes", " Yes")
    DEFAULT_NO_TOKENS = ("No", " no", "no", " No")

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        *,
        processor: str | None = None,
        model_class: str = "auto",
        prompt_template: str | None = None,
        image_token: str = "<image>",
        score_mode: str = "yes_no_probability",
        yes_tokens: Sequence[str] | None = None,
        no_tokens: Sequence[str] | None = None,
        trust_remote_code: bool = True,
        torch_dtype: str | None = "auto",
        device_map: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ):
        import torch

        processor = _none_if_blank(processor)
        device_map = _none_if_blank(device_map)
        torch_dtype = _none_if_blank(torch_dtype)
        self._torch = torch
        self.model_name = model
        self.device = device
        self.device_map = device_map
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.image_token = image_token
        self.score_mode = score_mode
        self.processor, self.model = _load_hf_vlm(
            model=model,
            processor=processor,
            model_class=model_class,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            model_kwargs=model_kwargs or {},
            processor_kwargs=processor_kwargs or {},
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.yes_token_ids = _first_token_ids(
            tokenizer, yes_tokens or self.DEFAULT_YES_TOKENS
        )
        self.no_token_ids = _first_token_ids(
            tokenizer, no_tokens or self.DEFAULT_NO_TOKENS
        )
        if not self.yes_token_ids or not self.no_token_ids:
            raise ValueError("yes_tokens and no_tokens must map to at least one token")

    def score_pairs(
        self, image_paths: Sequence[str | Path], texts: Sequence[str]
    ) -> list[float]:
        if len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have the same length")
        if not image_paths:
            return []

        images = [_load_rgb_image(path) for path in image_paths]
        prompts = [
            self.prompt_template.format(text=text, image=self.image_token)
            for text in texts
        ]
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs_to_device(inputs, self._input_device())
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = (
            inputs.get("attention_mask") if isinstance(inputs, dict) else None
        )
        return _score_yes_no_logits(
            outputs.logits,
            attention_mask=attention_mask,
            yes_token_ids=self.yes_token_ids,
            no_token_ids=self.no_token_ids,
            score_mode=self.score_mode,
        )

    def _input_device(self):
        if self.device_map is None:
            return self.device
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device


class VQAScoreAdapter:
    """VQAScore facade with pluggable model backends."""

    PA_LLAVA_PROMPT_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{image}\n"
        "Does this pathology image support the following question-answer "
        "statement?\n{text}\n"
        "Answer yes or no.<|eot_id|><|start_header_id|>assistant"
        "<|end_header_id|>\n\n"
    )
    BACKENDS = {
        "t2v": T2VVQAScoreBackend,
        "clip-flant5": T2VVQAScoreBackend,
        "clip_flant5": T2VVQAScoreBackend,
        "hf_yesno": HuggingFaceYesNoVLMAdapter,
        "huggingface_yesno": HuggingFaceYesNoVLMAdapter,
        "pallava": HuggingFaceYesNoVLMAdapter,
        "pa-llava": HuggingFaceYesNoVLMAdapter,
    }

    def __init__(
        self,
        model: str = "clip-flant5-xl",
        device: str = "cuda",
        *,
        backend: str = "t2v",
        **kwargs: Any,
    ):
        try:
            backend_cls = self.BACKENDS[backend]
        except KeyError as exc:
            options = ", ".join(sorted(self.BACKENDS))
            raise ValueError(
                f"Unknown VQAScore backend '{backend}'. Available: {options}"
            ) from exc

        if backend in {"pallava", "pa-llava"} and "prompt_template" not in kwargs:
            kwargs["prompt_template"] = self.PA_LLAVA_PROMPT_TEMPLATE
        self.backend = backend
        self.model_name = model
        self.device = device
        self._scorer = backend_cls(model=model, device=device, **kwargs)

    def score_pairs(
        self, image_paths: Sequence[str | Path], texts: Sequence[str]
    ) -> list[float]:
        return self._scorer.score_pairs(image_paths, texts)


def _tensor_scores_to_list(raw, n: int) -> list[float]:
    import torch

    if isinstance(raw, torch.Tensor):
        t = raw.detach().cpu()
        if t.ndim == 2 and t.shape[0] == t.shape[1] == n:
            return [float(t[i, i]) for i in range(n)]
        if t.ndim == 1 and t.numel() == n:
            return [float(t[i]) for i in range(n)]
        if t.numel() == 1:
            return [float(t.reshape(-1)[0])] * n
        flat = t.reshape(-1)
        return [float(flat[i]) for i in range(min(n, flat.numel()))]
    if isinstance(raw, (list, tuple)):
        out: list[float] = []
        for item in raw:
            if isinstance(item, torch.Tensor):
                out.append(float(item.detach().cpu().reshape(-1)[0]))
            else:
                out.append(float(item))
        return out[:n]
    return [float(raw)] * n


def _load_rgb_image(path_or_image):
    if isinstance(path_or_image, (str, bytes, Path)):
        return Image.open(path_or_image).convert("RGB")
    if hasattr(path_or_image, "convert"):
        return path_or_image.convert("RGB")
    return path_or_image


def _none_if_blank(value):
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _load_hf_vlm(
    *,
    model: str,
    processor: str | None,
    model_class: str,
    device: str,
    trust_remote_code: bool,
    torch_dtype: str | None,
    device_map: str | None,
    model_kwargs: dict[str, Any],
    processor_kwargs: dict[str, Any],
):
    import torch
    import transformers
    from transformers import AutoProcessor

    processor_obj = AutoProcessor.from_pretrained(
        processor or model,
        trust_remote_code=trust_remote_code,
        **processor_kwargs,
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        **model_kwargs,
    }
    dtype = _resolve_torch_dtype(torch, torch_dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model_obj = _load_hf_model_with_fallback(
        transformers=transformers,
        model=model,
        model_class=model_class,
        load_kwargs=load_kwargs,
    )
    if device_map is None and device not in {"auto", "none", None}:
        model_obj = model_obj.to(device)
    return processor_obj, model_obj.eval()


def _resolve_torch_dtype(torch, torch_dtype: str | None):
    if torch_dtype in {None, "none"}:
        return None
    if torch_dtype == "auto":
        return "auto"
    try:
        return getattr(torch, torch_dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}") from exc


def _load_hf_model_with_fallback(
    *,
    transformers,
    model: str,
    model_class: str,
    load_kwargs: dict[str, Any],
):
    if model_class != "auto":
        cls = getattr(transformers, model_class)
        return cls.from_pretrained(model, **load_kwargs)

    errors: list[str] = []
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model, **load_kwargs)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{class_name}: {exc}")
    details = " | ".join(errors) if errors else "no compatible AutoModel class found"
    raise RuntimeError(f"Could not load Hugging Face VLM '{model}': {details}")


def _move_inputs_to_device(inputs, device):
    if device in {"auto", "none", None}:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _first_token_ids(tokenizer, candidates: Sequence[str]) -> list[int]:
    special_ids = {
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    }
    out: list[int] = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not ids:
            continue
        token_id = None
        for item in ids:
            if item not in special_ids:
                token_id = int(item)
                break
        if token_id is None:
            token_id = int(ids[0])
        if token_id not in out:
            out.append(token_id)
    return out


def _score_yes_no_logits(
    logits,
    *,
    attention_mask,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
    score_mode: str,
) -> list[float]:
    import torch

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [batch, seq, vocab], got {logits.shape}"
        )
    batch_size, seq_len, _ = logits.shape
    if attention_mask is None:
        positions = torch.full(
            (batch_size,), seq_len - 1, dtype=torch.long, device=logits.device
        )
    else:
        mask = attention_mask.to(device=logits.device, dtype=torch.long)
        token_positions = torch.arange(seq_len, device=logits.device).expand_as(mask)
        positions = (mask * token_positions).max(dim=1).values

    batch_index = torch.arange(batch_size, device=logits.device)
    next_token_logits = logits[batch_index, positions, :]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    yes_logprob = _logsumexp_token_ids(log_probs, yes_token_ids)
    no_logprob = _logsumexp_token_ids(log_probs, no_token_ids)

    if score_mode == "yes_probability":
        scores = yes_logprob.exp()
    elif score_mode == "yes_no_probability":
        scores = torch.sigmoid(yes_logprob - no_logprob)
    elif score_mode == "logprob_margin":
        scores = yes_logprob - no_logprob
    else:
        raise ValueError(
            "score_mode must be one of: yes_probability, "
            "yes_no_probability, logprob_margin"
        )
    return [float(item) for item in scores.detach().cpu()]


def _logsumexp_token_ids(log_probs, token_ids: Sequence[int]):
    import torch

    if not token_ids:
        raise ValueError("token_ids must not be empty")
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=log_probs.device)
    return torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1)


class BlipStudentAdapter:
    """Frozen BLIP-VQA pretrained checkpoint for Gate 3 zero-shot scoring."""

    def __init__(
        self,
        checkpoint: str,
        med_config: str = "configs/med_config.json",
        image_size: int = 384,
        device: str = "cuda",
    ):
        import torch
        from models.blip_vqa import blip_vqa

        self._torch = torch
        self.device = device
        self.model = blip_vqa(
            pretrained=checkpoint,
            med_config=med_config,
            image_size=image_size,
            vit="base",
        )
        self.model = self.model.to(device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def answer_question(self, image, question: str) -> str:
        return self.answer_questions_batch([image], [question])[0]

    def answer_questions_batch(self, images: list, questions: list[str]) -> list[str]:
        if len(images) != len(questions):
            raise ValueError("images and questions must have the same length")
        if not images:
            return []
        tensors = []
        for image in images:
            if isinstance(image, (str, bytes)):
                image = Image.open(image).convert("RGB")
            tensors.append(self.preprocess(image))
        batch = self._torch.stack(tensors, dim=0).to(self.device)
        with self._torch.no_grad():
            answers = self.model(batch, questions, train=False, inference="generate")
        if isinstance(answers, list):
            return [str(a) for a in answers]
        return [str(answers)]
