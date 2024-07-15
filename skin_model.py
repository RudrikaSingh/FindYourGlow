import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

def get_season(img_path):
    """
    Predicts the season (skin tone category) of the given image.
    
    :param img_path: Path to the image file
    :return: Predicted season index
    """
    # Define the model
    model = models.resnet18(pretrained=True)
    num_classes = 4
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load saved state dictionary
    state_dict = torch.load('best_model_resnet_ALL.pth', map_location=torch.device('cpu'))

    # Create a new model with the correct architecture
    new_model = models.resnet18(pretrained=True)
    new_model.fc = nn.Linear(in_features, num_classes)
    new_model.load_state_dict(state_dict)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and transform the image
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Predict the season
    new_model.eval()
    with torch.no_grad():
        output = new_model(image)
    pred_index = output.argmax().item()
    return pred_index
