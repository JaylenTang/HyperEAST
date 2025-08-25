import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToPILImage

# Configuration (all parameters in one place)
class Config:
    DATASET_PATH = "/home/yubai03/yubai03/aJialin_Tang/Research/HyperEAST/lfam+hloss/data/IndianPine.mat"
    GT_PATH = "/home/yubai03/yubai03/aJialin_Tang/Research/HyperEAST/lfam+hloss/data/IndianPine.mat"
    MODEL_PATH = "./model_checkpoint.pth"
    OUTPUT_DIR = "./output/"
    INPUT_SIZE = 11
    BATCH_SIZE = 32
    NUM_CLASSES = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load data and ground truth
def load_data(data_path, gt_path):
    data = loadmat(data_path)['indian_pines_corrected']
    gt = loadmat(gt_path)['indian_pines_gt']
    return data.astype(np.float32), gt.astype(np.int64)

# Function to preprocess data
def preprocess_data(data, gt, input_size):
    margin = input_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    indices = np.nonzero(gt)
    patches = []
    labels = []

    for i, j in zip(*indices):
        patch = padded_data[i:i + input_size, j:j + input_size, :]
        patches.append(patch)
        labels.append(gt[i, j])

    patches = np.array(patches).transpose(0, 3, 1, 2)  # Convert to [N, C, H, W]
    labels = np.array(labels)

    dataset = TensorDataset(torch.tensor(patches, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    return loader

# Function to load model
def load_model(model_path, num_classes):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(Config.INPUT_SIZE * Config.INPUT_SIZE * 200, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(Config.DEVICE)
    model.eval()
    return model

# Function to generate prediction map
def generate_prediction_map(model, loader, gt_shape):
    predictions = np.zeros(gt_shape, dtype=int)
    k = 0

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred in preds:
                i, j = np.nonzero(gt_shape)[0][k], np.nonzero(gt_shape)[1][k]
                predictions[i, j] = pred + 1
                k += 1

    return predictions

# Function to visualize and save the prediction map
def visualize_map(prediction_map, gt_map, save_path):
    color_map = np.zeros((*prediction_map.shape, 3), dtype=np.uint8)
    colors = {
        0: [0, 0, 0],
        1: [147, 67, 46],
        2: [0, 0, 255],
        3: [255, 100, 0],
        4: [0, 255, 123],
        5: [164, 75, 155],
        6: [101, 174, 255],
        7: [118, 254, 172],
        8: [60, 91, 112],
        9: [255, 255, 0],
        10: [255, 255, 125],
        11: [255, 0, 255],
        12: [100, 0, 255],
        13: [0, 172, 254],
        14: [0, 255, 0],
        15: [171, 175, 80],
        16: [101, 193, 60]
    }

    for label, color in colors.items():
        color_map[prediction_map == label] = color

    plt.figure(figsize=(10, 10))
    plt.imshow(color_map)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Main function
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Data
    data, gt = load_data(Config.DATASET_PATH, Config.GT_PATH)

    # Step 2: Preprocess Data
    loader = preprocess_data(data, gt, Config.INPUT_SIZE)

    # Step 3: Load Model
    model = load_model(Config.MODEL_PATH, Config.NUM_CLASSES)

    # Step 4: Generate Prediction Map
    prediction_map = generate_prediction_map(model, loader, gt.shape)

    # Step 5: Visualize and Save Map
    visualize_map(prediction_map, gt, os.path.join(Config.OUTPUT_DIR, "prediction_map.png"))

if __name__ == "__main__":
    main()
