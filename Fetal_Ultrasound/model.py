from __future__ import annotations

from typing import Tuple


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies before creating models."
        ) from exc


def _layers():
    _require_tensorflow()
    from tensorflow.keras.layers import (
        Activation,
        Add,
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
        Multiply,
        Reshape,
        UpSampling2D,
        concatenate,
    )

    return {
        "Activation": Activation,
        "Add": Add,
        "BatchNormalization": BatchNormalization,
        "Conv2D": Conv2D,
        "Dense": Dense,
        "Dropout": Dropout,
        "GlobalAveragePooling2D": GlobalAveragePooling2D,
        "Input": Input,
        "MaxPooling2D": MaxPooling2D,
        "Multiply": Multiply,
        "Reshape": Reshape,
        "UpSampling2D": UpSampling2D,
        "concatenate": concatenate,
    }


def conv_block(input_tensor, num_filters: int, dropout_rate: float = 0.1):
    L = _layers()
    x = L["Conv2D"](num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(input_tensor)
    x = L["BatchNormalization"]()(x)
    x = L["Activation"]("relu")(x)
    x = L["Dropout"](dropout_rate)(x)

    x = L["Conv2D"](num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = L["BatchNormalization"]()(x)
    x = L["Activation"]("relu")(x)
    return x


def attention_block(x, g, inter_channel: int):
    L = _layers()
    theta_x = L["Conv2D"](inter_channel, (1, 1), padding="same")(x)
    phi_g = L["Conv2D"](inter_channel, (1, 1), padding="same")(g)
    add_xg = L["Add"]()([theta_x, phi_g])
    add_xg = L["Activation"]("relu")(add_xg)
    psi = L["Conv2D"](1, (1, 1), padding="same")(add_xg)
    psi = L["Activation"]("sigmoid")(psi)
    return L["Multiply"]()([x, psi])


def squeeze_excite_block(input_tensor, ratio: int = 8):
    L = _layers()
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)

    se = L["GlobalAveragePooling2D"]()(input_tensor)
    se = L["Reshape"](se_shape)(se)
    se = L["Dense"](max(filters // ratio, 1), activation="relu", kernel_initializer="he_normal")(se)
    se = L["Dense"](filters, activation="sigmoid", kernel_initializer="he_normal")(se)
    return L["Multiply"]()([input_tensor, se])


def build_attention_unet(
    input_size: Tuple[int, int, int] = (256, 256, 1),
    filters_base: int = 32,
):
    _require_tensorflow()
    from tensorflow.keras.models import Model

    L = _layers()
    inputs = L["Input"](input_size)

    c1 = conv_block(inputs, filters_base, dropout_rate=0.1)
    c1 = squeeze_excite_block(c1)
    p1 = L["MaxPooling2D"]((2, 2))(c1)

    c2 = conv_block(p1, filters_base * 2, dropout_rate=0.1)
    c2 = squeeze_excite_block(c2)
    p2 = L["MaxPooling2D"]((2, 2))(c2)

    c3 = conv_block(p2, filters_base * 4, dropout_rate=0.2)
    c3 = squeeze_excite_block(c3)
    p3 = L["MaxPooling2D"]((2, 2))(c3)

    c4 = conv_block(p3, filters_base * 8, dropout_rate=0.2)
    c4 = squeeze_excite_block(c4)
    p4 = L["MaxPooling2D"]((2, 2))(c4)

    c5 = conv_block(p4, filters_base * 16, dropout_rate=0.3)
    c5 = squeeze_excite_block(c5)

    u6 = L["UpSampling2D"]((2, 2))(c5)
    u6 = L["Conv2D"](filters_base * 8, (2, 2), padding="same", kernel_initializer="he_normal")(u6)
    a6 = attention_block(c4, u6, filters_base * 4)
    u6 = L["concatenate"]([u6, a6])
    c6 = conv_block(u6, filters_base * 8, dropout_rate=0.2)

    u7 = L["UpSampling2D"]((2, 2))(c6)
    u7 = L["Conv2D"](filters_base * 4, (2, 2), padding="same", kernel_initializer="he_normal")(u7)
    a7 = attention_block(c3, u7, filters_base * 2)
    u7 = L["concatenate"]([u7, a7])
    c7 = conv_block(u7, filters_base * 4, dropout_rate=0.2)

    u8 = L["UpSampling2D"]((2, 2))(c7)
    u8 = L["Conv2D"](filters_base * 2, (2, 2), padding="same", kernel_initializer="he_normal")(u8)
    a8 = attention_block(c2, u8, filters_base)
    u8 = L["concatenate"]([u8, a8])
    c8 = conv_block(u8, filters_base * 2, dropout_rate=0.1)

    u9 = L["UpSampling2D"]((2, 2))(c8)
    u9 = L["Conv2D"](filters_base, (2, 2), padding="same", kernel_initializer="he_normal")(u9)
    a9 = attention_block(c1, u9, max(filters_base // 2, 1))
    u9 = L["concatenate"]([u9, a9])
    c9 = conv_block(u9, filters_base, dropout_rate=0.1)

    outputs = L["Conv2D"](1, (1, 1), activation="sigmoid", name="output")(c9)
    return Model(inputs=[inputs], outputs=[outputs], name="AttentionUNet")


def build_unet(input_size: Tuple[int, int, int] = (256, 256, 1)):
    _require_tensorflow()
    from tensorflow.keras.models import Model

    L = _layers()
    inputs = L["Input"](input_size)

    c1 = conv_block(inputs, 32, dropout_rate=0.1)
    p1 = L["MaxPooling2D"]((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate=0.1)
    p2 = L["MaxPooling2D"]((2, 2))(c2)

    c3 = conv_block(p2, 128, dropout_rate=0.2)
    p3 = L["MaxPooling2D"]((2, 2))(c3)

    c4 = conv_block(p3, 256, dropout_rate=0.2)
    p4 = L["MaxPooling2D"]((2, 2))(c4)

    c5 = conv_block(p4, 512, dropout_rate=0.3)

    u6 = L["UpSampling2D"]((2, 2))(c5)
    u6 = L["concatenate"]([u6, c4])
    c6 = conv_block(u6, 256, dropout_rate=0.2)

    u7 = L["UpSampling2D"]((2, 2))(c6)
    u7 = L["concatenate"]([u7, c3])
    c7 = conv_block(u7, 128, dropout_rate=0.2)

    u8 = L["UpSampling2D"]((2, 2))(c7)
    u8 = L["concatenate"]([u8, c2])
    c8 = conv_block(u8, 64, dropout_rate=0.1)

    u9 = L["UpSampling2D"]((2, 2))(c8)
    u9 = L["concatenate"]([u9, c1])
    c9 = conv_block(u9, 32, dropout_rate=0.1)

    outputs = L["Conv2D"](1, (1, 1), activation="sigmoid", name="output")(c9)
    return Model(inputs=[inputs], outputs=[outputs], name="UNet")


def build_model(model_type: str = "unet", input_size: Tuple[int, int, int] = (256, 256, 1), filters_base: int = 32):
    model_type = model_type.lower().strip()
    if model_type == "attention_unet":
        return build_attention_unet(input_size=input_size, filters_base=filters_base)
    if model_type == "unet":
        return build_unet(input_size=input_size)
    raise ValueError(f"Unsupported model_type: {model_type}")
