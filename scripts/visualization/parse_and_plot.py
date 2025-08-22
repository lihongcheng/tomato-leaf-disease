import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log(file_path, model_name):
    """
    Parses a training log file to extract epoch, train loss, and validation loss.
    Handles different log formats.
    """
    data = []
    # Format 1: Epoch 1/100 | Train Loss: 1.2108 | Val Loss: 0.7322 | Val Acc: 0.9141
    regex1 = re.compile(r"Epoch (\d+)/\d+ \| Train Loss: (\d+\.\d+) \| Val Loss: (\d+\.\d+) \| Val Acc: (\d+\.\d+)")

    # Format 2: Epoch 1/100, train_loss: 0.9651, val_loss: 0.6766, val_accuracy: 0.4894
    regex2 = re.compile(r"Epoch (\d+)/\d+, train_loss: (\d+\.\d+), val_loss: (\d+\.\d+), val_accuracy: (\d+\.\d+)")

    with open(file_path, 'r') as f:
        for line in f:
            match = regex1.search(line)
            if not match:
                match = regex2.search(line)

            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                data.append([model_name, epoch, "Train Loss", train_loss])
                data.append([model_name, epoch, "Validation Loss", val_loss])

    return data

# --- Main Execution ---
log_files = [
    ("train_mobilenetv1.log", "MobileNetV1"),
    ("train_mobilenetv3_with_attention.log", "MobileNetV3-Attention"),
    ("train_ShuffleNetV2.output", "ShuffleNetV2"),
    ("train_efficientnetb0.output", "EfficientNet-B0"),
    ("train_mobilenetv3.output", "MobileNetV3-Small")
]

all_data = []
for log_file, model_name in log_files:
    try:
        file_path = f"/yourlogpath/{log_file}"
        model_data = parse_log(file_path, model_name)
        if model_data:
            all_data.extend(model_data)
        else:
            print(f"Warning: No data parsed from {log_file}")
    except FileNotFoundError:
        print(f"Error: Log file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {log_file}: {e}")


if not all_data:
    print("No data was parsed from any log file. Exiting.")
else:
    df = pd.DataFrame(all_data, columns=["Model", "Epoch", "Loss Type", "Loss"])

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plotting with Seaborn
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Loss",
        hue="Model",
        style="Loss Type",
        palette="tab10",
        linewidth=2.5
    )

    plt.title("Convergence of Training and Validation Loss for All Models", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(title="Model and Loss Type", fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure
    output_path = "/yoursvgpath/Fig5-4_training_loss_convergence_regenerated.svg"
    plt.savefig(output_path, format='svg')
    print(f"Plot saved to {output_path}")