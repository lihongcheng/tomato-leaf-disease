import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# Assuming data_process.py and evaluate_all_models.py are in the same directory
from data_process import DataProcessor, PlantDataset
from evaluate_all_models import load_model

# --- Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Note: Update this path if your checkpoints are located elsewhere
CHECKPOINT_PATH = "./checkpoints/MobileNetV3/best_checkpoint.pth"
MODEL_NAME = "MobileNetV3_small"
NUM_CLASSES = 11 
IMAGE_SIZE = (224, 224)
# Note: Update this path to your master CSV file
MASTER_CSV_PATH = "../dataset/metadata/master.csv"


# --- Grad-CAM Implementation ---
class GradCAM:
    """
    Minimal implementation of Grad-CAM for visualizing model decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            return None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, index=None):
        self.model.eval()
        output = self.model(x)
        
        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32, device=DEVICE)
        one_hot[0][index] = 1
        one_hot.requires_grad_(True)
        
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients.cpu().data.numpy()[0]
        target = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, IMAGE_SIZE)
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

def show_cam_on_image(img_np, mask):
    """Superimposes a CAM heatmap onto an image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img_np)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# --- Main Logic ---
def find_and_visualize_misclassified(model, loader, class_map, misclassifications_to_find):
    """
    Finds specific misclassified samples, generates Grad-CAM for them,
    and saves the result as a single SVG figure.
    """
    found_samples = {}
    inv_class_map = {v: k for k, v in class_map.items()}
    
    # The target layer for MobileNetV3-Small's features
    target_layer = model.features[-1]
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # Create a transform to un-normalize images for visualization
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )

    print("Searching for specified misclassification samples...")
    # Iterate through the dataset to find one example for each case
    for i in range(len(loader)):
        if len(found_samples) == len(misclassifications_to_find):
            break # Stop when all required samples are found

        img_tensor, label_idx = loader[i] # Get image and label directly from dataset
        img_tensor = img_tensor.to(DEVICE)

        output = model(img_tensor.unsqueeze(0))
        pred_idx = output.argmax(1).item()

        if pred_idx == label_idx:
            continue # Skip correctly classified samples

        true_label_name = inv_class_map[label_idx]
        pred_label_name = inv_class_map[pred_idx]
        
        key = (true_label_name, pred_label_name)
        if key in misclassifications_to_find and key not in found_samples:
            print(f"  Found: True='{key[0]}', Predicted='{key[1]}'")
            
            mask = grad_cam(img_tensor.unsqueeze(0), index=pred_idx)
            
            img_for_plot = inv_normalize(img_tensor.cpu())
            img_for_plot = img_for_plot.numpy().transpose(1, 2, 0)
            img_for_plot = np.clip(img_for_plot, 0, 1)

            cam_image = show_cam_on_image(img_for_plot, mask)
            
            # Store the original and the CAM image
            found_samples[key] = (img_for_plot, cam_image)

    if len(found_samples) != len(misclassifications_to_find):
        print("\nWarning: Could not find all specified misclassification examples.")
        print(f"Found {len(found_samples)} out of {len(misclassifications_to_find)}.")

    # --- Plotting ---
    if not found_samples:
        print("No samples found to plot. Exiting.")
        return

    # Ensure the order of plotting is the same as the requested order
    plot_order = [key for key in misclassifications_to_find if key in found_samples]
    
    fig, axs = plt.subplots(len(plot_order), 2, figsize=(8, 4 * len(plot_order)))
    fig.suptitle('Typical Misclassification Samples with Grad-CAM', fontsize=16)

    for i, key in enumerate(plot_order):
        true_label, pred_label = key
        orig_img, cam_img = found_samples[key]
        
        # Original Image
        axs[i, 0].imshow(orig_img)
        axs[i, 0].set_title(f'True: {true_label.replace("_", " ")}')
        axs[i, 0].axis('off')

        # Grad-CAM Image
        axs[i, 1].imshow(cam_img)
        axs[i, 1].set_title(f'Predicted: {pred_label.replace("_", " ")}')
        axs[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = 'Fig5-5_misclassification_samples.svg'
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"\nNew visualization saved to {output_filename}")
    plt.close()


if __name__ == '__main__':
    # 1. Load Model
    print("Loading model...")
    model = load_model(MODEL_NAME, NUM_CLASSES)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint.get('model_state', checkpoint.get('model', {})))
    model.to(DEVICE)
    model.eval()

    # 2. Load Dataset
    print("Loading dataset...")
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"Error: Master CSV file not found at {MASTER_CSV_PATH}")
        exit()
    processor = DataProcessor()
    master_df = processor.generate_metadata()
    test_df = master_df[master_df['split'] == 'test'].copy()
    test_dataset = PlantDataset(test_df, mode='val') 
    
    # 3. Define misclassifications to find based on confusion matrix
    misclassifications_to_find = [
        ('Healthy', 'Target_Spot'),
        ('Target_Spot', 'Bacterial_spot'),
        ('Late_blight', 'Early_blight')
    ]
    
    # 4. Find and visualize
    find_and_visualize_misclassified(model, test_dataset, test_dataset.class_map, misclassifications_to_find)
    print("Done.")
