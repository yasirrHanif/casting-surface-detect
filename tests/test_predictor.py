"""Checkpoint contract test with synthetic weights; doesn't need real training file."""

import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b3

from app.predictor import Predictor


def test_checkpoint_loads_and_predicts(tmp_path):

    # Create synthetic EfficientNet-B3 checkpoint
    model = efficientnet_b3(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )


    checkpoint = {

        "model_state": model.state_dict(),

        "model_name": "EfficientNet-B3",

        "class_to_idx": {
            "defective": 0,
            "normal": 1
        },

        "img_size": 224,

        "mean": [
            0.485,
            0.456,
            0.406
        ],

        "std": [
            0.229,
            0.224,
            0.225
        ],
    }


    path = tmp_path / "final_model.pt"


    torch.save(
        checkpoint,
        path
    )


    predictor = Predictor(
        str(path)
    )


    result = predictor.predict(
        Image.new(
            "RGB",
            (100, 100),
            "gray"
        )
    )


    assert result["predicted_class"] in {
        "defective",
        "normal"
    }


    assert 0 <= result["confidence_score"] <= 1


    assert abs(
        sum(result["probabilities"].values()) - 1
    ) < 1e-5