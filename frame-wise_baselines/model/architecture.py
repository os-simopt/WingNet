# model/architecture.py

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetV2B1, MobileNetV2

# The supplied checkpoint matches Keras ConvNeXtSmall, not ConvNeXtBase.
try:
    from tensorflow.keras.applications import ConvNeXtSmall
except ImportError:
    ConvNeXtSmall = None

import config


def _stem(inp):
    """Convert single-channel Bayer-like input to 3-channel image."""
    # Bayer-like pooling (kept for compatibility with original notebook)
    x = AveragePooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding="same",
        name="serving_default_input_1",
    )(inp)

    # Convert grayscale to RGB (1 channel -> 3 channels)
    x = Conv2D(3, (1, 1), padding="same", use_bias=True)(x)
    return x


def _head(x, num_classes, weight_decay, dropout):
    """Classification head: GAP -> Dense -> Dropout -> Dense(softmax)."""
    x = GlobalAveragePooling2D()(x)
    x = Dense(
        x.shape[-1] // 2,
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = Dropout(dropout)(x)
    x = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(weight_decay),
        name="StatefulPartitionedCall",
    )(x)
    return x


def _build_backbone(x, backbone_name: str):
    """
    Attach the chosen CNN backbone (effnetv2b1, mobilenetv2, convnextsmall)
    to the stem output.
    """
    input_shape_3ch = (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)

    if backbone_name == "effnetv2b1":
        backbone = EfficientNetV2B1(
            input_shape=input_shape_3ch,
            include_top=False,
            weights=None,
        )
        return backbone(x)

    if backbone_name == "mobilenetv2":
        backbone = MobileNetV2(
            input_shape=input_shape_3ch,
            include_top=False,
            weights=None,
        )
        return backbone(x)

    if backbone_name == "convnextsmall":
        if ConvNeXtSmall is None:
            raise ImportError(
                "ConvNeXtSmall is not available in this TensorFlow/Keras version."
            )
        backbone = ConvNeXtSmall(
            input_shape=input_shape_3ch,
            include_top=False,
            weights=None,
        )
        return backbone(x)

    raise ValueError(f"Unknown backbone '{backbone_name}'")


def create_model() -> Model:
    """
    Build the CNN classifier using the backbone specified in config.BACKBONE.

    The overall structure matches the original notebook:
    - Input: (128, 128, 1)
    - Stem: pooling + 1x1 conv to 3 channels
    - Backbone: EfficientNetV2B1 / MobileNetV2 / ConvNeXtSmall
    - Head: GAP -> Dense -> Dropout -> Dense(softmax)
    """
    inp = Input(shape=config.INPUT_SHAPE)

    # Stem (Bayer-like pooling + grayscale->RGB)
    x = _stem(inp)

    # Backbone chosen via config.BACKBONE
    x = _build_backbone(x, config.BACKBONE)

    # Classification head
    out = _head(x, config.NUM_CLASSES, config.WEIGHT_DECAY, config.DROPOUT)

    model = Model(inp, out)
    return model
