import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
# Assuming dataset.py is in the same directory and contains the get_loaders function
from dataset import get_loaders # Explicitly import the function
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

# --- CIFAR-10 Class Definitions ---
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# =================================================================
#  RESNET ARCHITECTURE DEFINITION
# =================================================================

class BasicBlock(nn.Module):
    """
    The fundamental building block for ResNet-18/34, consisting of two 3x3 convolutions
    and a skip connection to handle the residual mapping.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1st 3x3 Conv Layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 2nd 3x3 Conv Layer
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip Connection (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Match dimensions if stride changes (downsampling) or channels increase
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Convolution 1 -> BN -> ReLU
        out = self.bn1(self.conv1(x))
        out = nn.ReLU()(out)
        
        # Convolution 2 -> BN
        out = self.bn2(self.conv2(out))
        
        # Add the original input (shortcut) to the output
        out += self.shortcut(x) 
        
        # Final ReLU
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    """
    A custom ResNet tailored for the 32x32 input size of CIFAR-10 (like ResNet-20).
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16 # Start with 16 channels, common for CIFAR-ResNet

        # 1. Initial Convolution (3x3, C-3 to C-16, maintains 32x32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. ResNet Stages (S1, S2, S3)
        # Stride=1, maintains 32x32
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1) 
        # Stride=2, downsamples to 16x16
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2) 
        # Stride=2, downsamples to 8x8
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) 
        
        # 3. Final Linear Layer
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # The first block in a stage handles the downsampling/channel increase
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. Initial Conv -> BN -> ReLU
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        
        # 2. Stages
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        
        # 3. Global Average Pooling (8x8 -> 1x1)
        # The pooling kernel size matches the final spatial size (8)
        out = nn.AvgPool2d(kernel_size=out.size(2))(out) 
        
        # 4. Flatten and Linear
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# =================================================================
#  MODEL INITIALIZATION & UTILITIES
# =================================================================

def initialize_model(model_type, device):
    """Initializes a model based on the model_type argument for CIFAR-10."""
    
    num_classes = 10
    
    if model_type == 'simple_linear':
        print(f"Initializing Simple Linear Model...")
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, num_classes)
        ).to(device)
    
    elif model_type == 'simple_cnn':
        print(f"Initializing Simple CNN Model...")
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3), 
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, num_classes)
        ).to(device)
    
    elif model_type == 'resnet20':
        # ResNet-20 for CIFAR uses [3, 3, 3] blocks for layers 1, 2, and 3.
        print(f"Initializing ResNet-20 Model (Custom for 32x32 input)...")
        model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes).to(device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return model


def visualize_mispredictions(mispredictions_list, max_display=5):
    """
    Draws a grid of mispredicted images with their true and predicted labels.
    """
    num_display = min(len(mispredictions_list), max_display)
    
    if num_display == 0:
        print("No mispredictions collected to visualize.")
        return
        
    cols = 5
    rows = int(np.ceil(num_display / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    if rows == 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()
    
    print(f"\n🖼️ Displaying {num_display} Mispredicted Samples...")

    for i in range(num_display):
        sample = mispredictions_list[i]
        
        image_tensor = sample['image'] 
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)

        ax = axes[i]
        ax.imshow(image_np)
        
        true_label_name = CIFAR10_CLASSES[sample['true']]
        pred_label_name = CIFAR10_CLASSES[sample['predicted']]
        
        title = f"GT: {true_label_name}\nPred: {pred_label_name}"
        ax.set_title(title, color='red', fontsize=10)
        ax.axis('off')

    for j in range(num_display, len(axes)):
        if j < len(axes):
            fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, criterion, optimizer, device, args, run):
    """Handles the training loop."""
    print(f"\n🔥 Starting Training for {args.num_epochs} epochs...")
    model.train()
    
    for i in range(args.num_epochs): 
        epoch_loss = 0
        for img, label in tqdm(train_loader, desc=f"Epoch {i+1}/{args.num_epochs}"): 
            img = img.to(device)
            label = label.to(device)
            
            # Forward pass
            logits = model(img)
            loss = criterion(logits, label)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            run.log({"batch_loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {i+1} loss: {avg_epoch_loss:.4f}")
        run.log({"avg_epoch_loss": avg_epoch_loss, "epoch": i+1})
    
    # Save the model after training
    torch.save(model.state_dict(), args.model_save)
    print(f"✅ Model weights saved to {args.model_save}")


def check_accuracy(model, data_loader, criterion, device, mode_name="Test"):
    """
    Calculates the loss and accuracy for a given data loader.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for img, label in tqdm(data_loader, desc=f"Checking {mode_name} Acc"):
            img = img.to(device)
            label = label.to(device)
            
            logits = model(img)
            loss = criterion(logits, label)
            total_loss += loss.item() * img.size(0) 

            _, predicted = torch.max(logits, 1) 
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples * 100
    
    print(f"\n{mode_name} Set Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    return avg_loss, accuracy


def evaluate_model(model, train_loader, test_loader, criterion, device, args):
    """
    Handles evaluation, checking accuracy on both train and test sets, 
    and visualizing 5 mispredictions from the test set.
    """
    print("\n🧠 Starting Evaluation (Checking Train and Test Accuracy)...")
    
    # 1. Load the trained model state
    try:
        model.load_state_dict(torch.load(args.model_save, map_location=device))
        print(f"✅ Model weights loaded from {args.model_save}")
    except FileNotFoundError:
        print(f"❌ Model file not found at {args.model_save}. Cannot evaluate.")
        return

    model.to(device)
    
    # 2. Check Accuracy on Training Set
    train_loss, train_acc = check_accuracy(
        model, train_loader, criterion, device, mode_name="Training"
    )
    
    # 3. Full Test Set Pass and Misprediction Collection
    
    mispredicted_samples = []
    
    test_loss_sum = 0.0
    test_correct_sum = 0
    test_total_sum = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Full Test Set Pass"):
            
            img_inf = img.to(device)
            label_inf = label.to(device)
            
            logits = model(img_inf)
            loss = criterion(logits, label_inf)

            # Update metrics for the full run
            test_loss_sum += loss.item() * img_inf.size(0)
            _, predicted = torch.max(logits, 1)
            test_correct_sum += (predicted == label_inf).sum().item()
            test_total_sum += label_inf.size(0)
            
            # Store Mispredictions (Only if we still need more)
            if len(mispredicted_samples) < 5:
                mispredictions_mask = (predicted != label_inf)
                
                if mispredictions_mask.any():
                    mismatched_indices = torch.where(mispredictions_mask)[0]

                    for idx in mismatched_indices:
                        if len(mispredicted_samples) < 5:
                            mispredicted_samples.append({
                                'image': img[idx].cpu(),
                                'true': label[idx].item(),
                                'predicted': predicted[idx].item()
                            })
                        
    # Calculate final Test Metrics from the full run
    test_avg_loss = test_loss_sum / test_total_sum
    test_acc = test_correct_sum / test_total_sum * 100
    
    print("\nTest (Full Pass) Set Results:")
    print(f"  Loss: {test_avg_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}% ({test_correct_sum}/{test_total_sum})")

    # 4. Final Summary
    print("\n--- Summary of Results ---")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:     {test_acc:.2f}%")
    if train_acc > test_acc + 5.0:
        print("💡 **Warning:** High potential for **Overfitting** (Train Acc >> Test Acc).")
    print("--------------------------")
    
    # 5. Visualize the collected mispredictions
    visualize_mispredictions(mispredicted_samples, max_display=5)


def main(args):
    """
    Main function to initialize environment and dispatch to training or evaluation.
    """
    print("🚀 Starting initialization...")
    
    # 1. Device Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware device selected: {device}")
    
    # 2. Data Loading
    PKL_PATH = "./cifar10.pkl" 
    
    try:
        data_loaders = get_loaders(
            pkl_path=PKL_PATH,
            batch_size=args.batch_size, 
            num_workers=1
        )
        train_loader = data_loaders['train']
        test_loader  = data_loaders['test']
        print(f"✅ Data loaded.")
    except Exception as e:
        print(f"❌ Error during data loading. Check PKL_PATH and dataset.py: {e}")
        return

    # 3. Model Initialization
    model = initialize_model(args.model_type, device)
    
    # 4. Optimizer and Loss Function Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Mode Dispatch
    if args.mode == 'train':
        run = wandb.init(
            entity="chihuahua",
            project="Ciphar10_Final_Project",
            name=args.exp_name,
            config=vars(args)
        )
        train_model(model, train_loader, criterion, optimizer, device, args, run)
        run.finish()
        
    elif args.mode == 'test':
        evaluate_model(model, train_loader, test_loader, criterion, device, args)
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Must be 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Classifier: Train or Test.")

    # Main Mode Argument
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Operation mode: "train" for training, "test" for evaluation.')

    # Core Hyperparameters (mostly for training)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of samples per batch to load (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of full passes over the training data (default: 10)')

    # Model and Environment Parameters
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'simple_linear', 'resnet20'], # <-- UPDATED
                        help='Type of model architecture to use (default: simple_cnn)')
    parser.add_argument('--model_save', type=str, default='cifar10_best.pth',
                        help='Path to save/load model weights (default: cifar10_best.pth).')
    parser.add_argument('--exp_name', type=str, default='cnn_baseline',
                        help='Experiment name for W&B logging (default: cnn_baseline).')
    
    args = parser.parse_args()
    main(args)