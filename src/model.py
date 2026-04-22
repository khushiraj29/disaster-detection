"""
model.py
--------
Defines the MobileNetV2-based transfer learning model for the
Lightweight CNN-Based Disaster Detection Framework.

Architecture overview:
  ┌─────────────────────────────────────────────────────────┐
  │  Input (224 × 224 × 3)                                  │
  │  MobileNetV2 backbone (pretrained on ImageNet, frozen)  │
  │  GlobalAveragePooling2D                                 │
  │  Dense(256, relu) + Dropout(0.4)                        │
  │  Dense(128, relu) + Dropout(0.3)                        │
  │  Dense(5, softmax) — output class probabilities         │
  └─────────────────────────────────────────────────────────┘

The base MobileNetV2 layers are frozen during Phase 1 training.
For Phase 2 (fine-tuning), the top N layers are unfrozen and
trained with a very small learning rate.
"""

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES  = 5
IMAGE_SIZE   = (224, 224)
INPUT_SHAPE  = (*IMAGE_SIZE, 3)


def build_model(num_classes: int = NUM_CLASSES,
                dropout_rate_1: float = 0.40,
                dropout_rate_2: float = 0.30,
                l2_reg: float = 1e-4) -> models.Model:
    """
    Builds and returns the transfer-learning CNN model.

    Design decisions:
    - MobileNetV2 is chosen for its depthwise-separable convolutions,
      which dramatically reduce parameters while preserving accuracy —
      ideal for resource-constrained deployment.
    - `include_top=False` removes the ImageNet classification head so we
      can attach our own 5-class softmax head.
    - L2 regularisation on Dense layers prevents over-fitting on small
      datasets.

    Parameters
    ----------
    num_classes    : Number of output classes (default 5).
    dropout_rate_1 : Dropout rate after the first Dense layer.
    dropout_rate_2 : Dropout rate after the second Dense layer.
    l2_reg         : L2 regularisation coefficient for Dense layers.

    Returns
    -------
    model : Compiled Keras Model ready for training.
    """

    # ── 1. Load MobileNetV2 backbone ──────────────────────────────────────
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,          # Remove the default ImageNet head
        weights="imagenet",         # Use ImageNet pretrained weights
    )

    # Freeze ALL base layers — we only train the custom head in Phase 1
    base_model.trainable = False

    # ── 2. Build the custom classification head ────────────────────────────
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")

    # Pass inputs through the frozen backbone (training=False keeps
    # BatchNorm layers in inference mode even during training)
    x = base_model(inputs, training=False)

    # Reduce spatial dimensions to a 1320-D feature vector
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # First dense block: learn high-level combinations of backbone features
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_256",
    )(x)
    x = layers.Dropout(dropout_rate_1, name="dropout_1")(x)

    # Second dense block: further specialise for disaster categories
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_128",
    )(x)
    x = layers.Dropout(dropout_rate_2, name="dropout_2")(x)

    # Output layer: softmax produces a probability distribution over classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    model = models.Model(inputs, outputs, name="DisasterDetector_MobileNetV2")

    return model


def compile_model(model: models.Model, learning_rate: float = 1e-3):
    """
    Compiles the model with Adam optimiser and categorical cross-entropy loss.

    Parameters
    ----------
    model         : Keras Model returned by build_model().
    learning_rate : Initial learning rate for Adam (default 1e-3).
    """
    import tensorflow as tf

    model.compile(
        # Adam adapts learning rates per-parameter — efficient on sparse data
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Categorical cross-entropy is the standard loss for multi-class
        # one-hot encoded targets
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: models.Model,
                        num_layers_to_unfreeze: int = 30,
                        fine_tune_lr: float = 1e-5):
    """
    Phase 2 fine-tuning: unfreezes the top `num_layers_to_unfreeze` layers
    of the MobileNetV2 backbone and recompiles with a very small LR.

    Fine-tuning allows the backbone to adapt its high-level features to the
    disaster domain while the very small learning rate avoids catastrophic
    forgetting of the ImageNet representations.

    Parameters
    ----------
    model                  : Trained model from Phase 1.
    num_layers_to_unfreeze : Number of top backbone layers to unfreeze.
    fine_tune_lr           : Learning rate for fine-tuning (default 1e-5).
    """
    import tensorflow as tf

    # Find the MobileNetV2 sub-model inside the functional model
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True

    # Freeze everything except the top N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Recompile with a reduced learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Fine-tuning enabled: top {num_layers_to_unfreeze} backbone layers unfrozen.")
    return model


def model_summary(model: models.Model) -> None:
    """Prints model summary with parameter counts."""
    model.summary()
    total     = model.count_params()
    trainable = sum(
        p.numpy().size
        for p in model.trainable_weights
    )
    print(f"\nTotal parameters   : {total:,}")
    print(f"Trainable params   : {trainable:,}")
    print(f"Non-trainable params: {total - trainable:,}")
