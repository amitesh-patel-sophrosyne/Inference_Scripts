from transformers import PretrainedConfig
import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    PreTrainedModel
)


class ECGDinoConfig(PretrainedConfig):

    model_type = "ecg_dino"

    def __init__(
        self,
        image_model_name="facebook/dinov2-small",
        num_features=6,
        num_labels=2,
        feature_hidden_dim=32,
        feature_output_dim=128,
        classifier_hidden_dim=256,
        dropout=0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_model_name = image_model_name
        self.num_features = num_features
        self.num_labels = num_labels

        self.feature_hidden_dim = feature_hidden_dim
        self.feature_output_dim = feature_output_dim

        self.classifier_hidden_dim = classifier_hidden_dim
        self.dropout = dropout


class ECGDinoModel(PreTrainedModel):

    config_class = ECGDinoConfig
    base_model_prefix = "ecg_dino"

    def __init__(self, config):
        super().__init__(config)

        self.image_model = AutoModel.from_pretrained(
            config.image_model_name
        )

        hidden_dim = self.image_model.config.hidden_size

        self.feature_encoder = nn.Sequential(
            nn.Linear(
                config.num_features,
                config.feature_hidden_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(
                config.feature_hidden_dim
            ),

            nn.Linear(
                config.feature_hidden_dim,
                config.feature_output_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(
                config.feature_output_dim
            )
        )

        self.image_norm = nn.LayerNorm(
            hidden_dim
        )

        self.feature_norm = nn.LayerNorm(
            config.feature_output_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim + config.feature_output_dim,
                config.classifier_hidden_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(
                config.classifier_hidden_dim
            ),
            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.classifier_hidden_dim,
                config.num_labels
            )
        )

        self.post_init()

    def forward(
        self,
        pixel_values,
        ecg_features,
        labels=None
    ):

        outputs = self.image_model(
            pixel_values=pixel_values
        )

        image_features = outputs.pooler_output

        image_features = self.image_norm(
            image_features
        )

        feature_features = self.feature_encoder(
            ecg_features.float()
        )

        feature_features = self.feature_norm(
            feature_features
        )

        fused = torch.cat(
            [
                image_features,
                feature_features
            ],
            dim=1
        )

        logits = self.classifier(
            fused
        )

        loss = None

        if labels is not None:

            loss_fn = nn.BCEWithLogitsLoss()

            loss = loss_fn(
                logits,
                labels.float()
            )

        return {
            "logits": logits
        }
