# Plant Disease Detection using Deep Learning

This project is an end-to-end deep learning application that identifies plant diseases from leaf images. It uses a fine-tuned **ResNet-18** model trained on the **PlantVillage** dataset and includes a fully custom, minimal, pastel-themed Streamlit frontend designed for clarity and smooth interaction.

## Features

**Model**
Trained on 38 plant disease classes from the PlantVillage dataset.
Uses transfer learning with ResNet-18.
Achieves high validation accuracy on structured dataset images.
Supports inference on user-uploaded leaf photos.

**Frontend**
Custom Streamlit interface with:
a pastel green theme
soft glass-style hero section
clean upload card
animated result card
visible leaf texture background
all support containers/white blocks removed for a minimal, polished look

**Pipeline**
Training, validation, and model saving
Real-time inference using Streamlit
Deployable locally or on cloud platforms such as Streamlit Cloud, Render, or Hugging Face Spaces

## How It Works

1. Upload a leaf image (JPG/PNG).
2. The model preprocesses the image and runs inference.
3. The interface displays the predicted disease and confidence score.
4. The uploaded image is shown for reference.

## Tech Stack

Python
PyTorch
Torchvision
Streamlit
PlantVillage Dataset
ResNet-18 Transfer Learning

## Running the Project

1. Install dependencies:

```
pip install streamlit torch torchvision pillow
```

2. Place your trained model file:

```
best_model.pth
```

in the same folder as `app.py`.

3. Run the application:

```
streamlit run app.py
```

4. Open the local URL shown in your terminal.

## Dataset

The project uses the **New Plant Diseases Dataset (Augmented)** version of PlantVillage, which contains 38 classes across healthy and diseased categories. Images are high-quality and lab-captured, making the model highly accurate in structured environments.
