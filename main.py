import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, AutoImageProcessor

# Load model and processor
model_name = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True).to(device)


# Function to convert grayscale to RGB
def convert_to_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# Function to apply attention heatmap
def apply_attention_heatmap(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)

    attentions = outputs.attentions  # Extract attention layers
    last_attention = attentions[-1].squeeze(0).mean(dim=0)  # Average across heads

    # Resize the attention heatmap to match image size
    heatmap = last_attention.mean(dim=0).detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    image_np = np.array(image.convert("RGB"))
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    return superimposed_img


# Function to predict pneumonia
def predict(image):
    try:
        image = convert_to_rgb(image)  # Ensure it's RGB
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = torch.argmax(outputs.logits).item()

        return predicted_class_idx
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None


# Streamlit App
def main():
    st.title("Pneumonia Detection from Chest X-ray")
    st.write("Upload a chest X-ray image to detect pneumonia.")

    uploaded_image = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Pneumonia"):
            prediction_idx = predict(image)
            if prediction_idx is not None:
                if prediction_idx == 1:
                    st.error("Pneumonia Detected")
                    gradcam_image = apply_attention_heatmap(image, model)
                    st.image(gradcam_image, caption="Attention Heatmap", use_column_width=True)
                else:
                    st.success("No Pneumonia Detected")


if __name__ == "__main__":
    main()
