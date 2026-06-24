from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


from backend.app.services.model_loader import load_model, load_best_model_config
from backend.app.services.preprocessing import load_image_as_array


def find_last_conv_layer(model):
    """
    Finds the last convolution-like layer in the full model.
    This works when EfficientNetV2B0 layers are expanded into the main model.
    """
    conv_layer_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
    )

    for layer in reversed(model.layers):
        if isinstance(layer, conv_layer_types):
            print(f"Using last conv layer for Grad-CAM: {layer.name}")
            return layer.name

    print("\nCould not find direct Conv2D/DepthwiseConv2D layer.")
    print("Last 80 model layers are:")

    for layer in model.layers[-80:]:
        print(layer.name, type(layer))

    raise ValueError("Could not find convolution layer in full model.")


def make_gradcam_heatmap(image_array, pred_index=None):
    model = load_model()

    last_conv_layer_name = find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output,
        ],
    )

    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None. Grad-CAM could not compute gradients.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if float(max_value.numpy()) != 0:
        heatmap = heatmap / max_value

    return heatmap.numpy(), int(pred_index), float(predictions[0][pred_index])


def save_gradcam_overlay(image_path, output_path, alpha=0.4):
    config = load_best_model_config()
    image_size = tuple(config["image_size"])

    image_array = load_image_as_array(image_path, image_size=image_size)

    heatmap, pred_index, confidence = make_gradcam_heatmap(image_array)

    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize(image_size)

    heatmap_image = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_image = heatmap_image.resize(original_image.size)

    heatmap_array = np.array(heatmap_image)

    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_array]
    jet_heatmap = np.uint8(255 * jet_heatmap)

    original_array = np.array(original_image)

    overlay = np.uint8(
        original_array * (1 - alpha) + jet_heatmap * alpha
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_path)

    return {
        "gradcam_path": str(output_path),
        "predicted_index": pred_index,
        "confidence": round(confidence, 6),
    }
