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
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            answers = self.model(tensor, question, train=False, inference="generate")
        return answers[0] if isinstance(answers, list) else str(answers)
