import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torch.nn as nn
import torch.nn.functional as F

# Define SimpleCNN class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 256 * 256, 1)  # Adjust based on input image size and network architecture

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 256 * 256)  # Adjust based on input image size and network architecture
        x = self.fc1(x)
        return x

# Load the PyTorch model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit the model input size
    transforms.ToTensor(),           # Convert image to tensor
])

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = Image.open(image).convert('RGB')  # Convert image to RGB
    image = transform(image).unsqueeze(0)     # Apply transformations and add batch dimension
    return image

# Define the Streamlit app
def main():
    st.title('Infrared Image Intensity Prediction')

    # Upload image
    uploaded_file = st.file_uploader("Upload Infrared Image", type=['jpg', 'jpeg'])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = preprocess_image(uploaded_file)

        # Make prediction
        with torch.no_grad():
            output = model(image)

        # Display prediction result
        st.write(f"Predicted Intensity: {output}")

if __name__ == '__main__':
    main()
