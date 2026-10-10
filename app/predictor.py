"""Load the exact EfficientNet-B3 checkpoint exported by the Kaggle notebook."""

from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b3


class Predictor:
    """Thread-safe for read-only inference after initialization."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:

        file = Path(checkpoint_path)

        if not file.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{file}'. "
                "Copy Kaggle outputs/final_model.pt into models/final_model.pt."
            )


        # Load checkpoint
        checkpoint = torch.load(
            file,
            map_location="cpu",
            weights_only=True
        )


        if (
            not isinstance(checkpoint, dict)
            or "model_state" not in checkpoint
        ):
            raise ValueError(
                "Expected final_model.pt checkpoint dictionary."
            )


        # Verify class mapping

        self.class_to_idx = checkpoint.get(
            "class_to_idx"
        )


        if self.class_to_idx != {
            "defective": 0,
            "normal": 1
        }:
            raise ValueError(
                f"Unexpected class mapping: {self.class_to_idx!r}"
            )


        # Verify architecture

        if checkpoint.get("model_name") != "EfficientNet-B3":

            raise ValueError(
                "Unexpected architecture. Expected EfficientNet-B3."
            )


        # Load preprocessing metadata

        size = int(
            checkpoint["img_size"]
        )

        mean = checkpoint["mean"]

        std = checkpoint["std"]


        if (
            not 64 <= size <= 1024
            or len(mean) != 3
            or len(std) != 3
        ):
            raise ValueError(
                "Invalid preprocessing metadata."
            )


        # Device

        self.device = torch.device(device)


        # Build EfficientNet-B3 architecture

        self.model = efficientnet_b3(
            weights=None
        )


        in_features = (
            self.model
            .classifier[1]
            .in_features
        )


        self.model.classifier[1] = nn.Linear(
            in_features,
            len(self.class_to_idx)
        )


        # Load trained weights

        self.model.load_state_dict(
            checkpoint["model_state"],
            strict=True
        )


        self.model.to(
            self.device
        )

        self.model.eval()


        self.idx_to_class = {
            v: k
            for k, v in self.class_to_idx.items()
        }



        # Same preprocessing as validation/test

        self.transform = transforms.Compose([

            transforms.Resize(
                (size, size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean,
                std
            )

        ])



    def predict(self, image: Image.Image) -> dict:

        start = perf_counter()


        batch = (
            self.transform(
                image.convert("RGB")
            )
            .unsqueeze(0)
            .to(self.device)
        )


        with torch.inference_mode():

            outputs = self.model(batch)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]


        index = int(
            torch.argmax(probabilities).item()
        )


        scores = probabilities.cpu().tolist()


        return {

            "predicted_class":
                self.idx_to_class[index],


            "confidence_score":
                round(
                    float(scores[index]),
                    6
                ),


            "probabilities": {

                label:
                    round(
                        float(scores[idx]),
                        6
                    )

                for label, idx
                in self.class_to_idx.items()

            },


            "inference_time_ms":
                round(
                    (perf_counter() - start) * 1000,
                    2
                )
        }