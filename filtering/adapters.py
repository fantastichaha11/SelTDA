"""Real-model adapters for CLIP-ITM and BLIP Student scoring."""

from __future__ import annotations

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
