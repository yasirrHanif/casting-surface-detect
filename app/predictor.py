"""Load the exact EfficientNet-B3 checkpoint exported by the Kaggle notebook."""

from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b3


EXPECTED_CLASSES = {
    "defective": 0,
    "normal": 1,
}


class Predictor:
    """
    Thread-safe inference wrapper for EfficientNet-B3.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu"
    ) -> None:

        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_file}"
            )


        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=True
        )


        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid checkpoint format. Expected dictionary."
            )


        required_keys = [
            "model_state",
            "model_name",
            "class_to_idx",
            "img_size",
            "mean",
            "std",
        ]


        missing = [
            key for key in required_keys
            if key not in checkpoint
        ]


        if missing:
            raise ValueError(
                f"Checkpoint missing keys: {missing}"
            )


        # Class mapping check

        self.class_to_idx = checkpoint["class_to_idx"]


        if self.class_to_idx != EXPECTED_CLASSES:
            raise ValueError(
                f"Unexpected class mapping: {self.class_to_idx}"
            )


        # Architecture check

        if checkpoint["model_name"] != "EfficientNet-B3":
            raise ValueError(
                f"Expected EfficientNet-B3, "
                f"found {checkpoint['model_name']}"
            )


        # Preprocessing metadata

        img_size = int(
            checkpoint["img_size"]
        )

        mean = checkpoint["mean"]
        std = checkpoint["std"]


        if not (
            64 <= img_size <= 1024
            and len(mean) == 3
            and len(std) == 3
        ):
            raise ValueError(
                "Invalid preprocessing metadata."
            )


        self.device = torch.device(device)


        # Build model

        self.model = efficientnet_b3(
            weights=None
        )


        input_features = (
            self.model
            .classifier[1]
            .in_features
        )


        self.model.classifier[1] = nn.Linear(
            input_features,
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
            index: label
            for label, index
            in self.class_to_idx.items()
        }


        # Same transform as validation/test

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean,
                    std
                ),
            ]
        )


    def predict(
        self,
        image: Image.Image
    ) -> dict:

        start_time = perf_counter()


        tensor = (
            self.transform(
                image.convert("RGB")
            )
            .unsqueeze(0)
            .to(self.device)
        )


        with torch.inference_mode():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


        predicted_index = int(
            torch.argmax(probabilities)
        )


        scores = probabilities.cpu().tolist()


        return {

            "predicted_class":
                self.idx_to_class[predicted_index],


            "confidence_score":
                round(
                    float(scores[predicted_index]),
                    6
                ),


            "probabilities":
                {
                    label:
                        round(
                            float(scores[index]),
                            6
                        )
                    for label, index
                    in self.class_to_idx.items()
                },


            "inference_time_ms":
                round(
                    (perf_counter() - start_time)
                    * 1000,
                    2
                )
        }