import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from BitNetMCU import BitLinear, BitConv2d, Activation
import time
import random
import argparse
import yaml
from torchsummary import summary
import importlib
from models import MaskingLayer
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# Seed helper function
def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

# StratifiedKFold
def build_imagefolder_kfold_datasets(hyperparameters, fold_index):
    root_dir = hyperparameters["kfold_folder"]
    k_folds = hyperparameters.get("k_folds", 5)
    seed = hyperparameters.get("seed", 1234)

    mean = hyperparameters.get("mean", [0.1307])
    std = hyperparameters.get("std", [0.3081])

    if isinstance(mean, (float, int)):
        mean = (float(mean),)
    else:
        mean = tuple(mean)

    if isinstance(std, (float, int)):
        std = (float(std),)
    else:
        std = tuple(std)

    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.normalize(mean, std),
    ])

    aug_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
        transforms.RandomAffine(
            degrees=hyperparameters["rotation2"],
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=40.0, sigma=4.0)
        ], p=hyperparameters["elastictransformprobability"]),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.normalize(mean, std),
    ])

    base_dataset = datasets.ImageFolder(root=root_dir, transform=base_transform)
    aug_dataset = datasets.ImageFolder(root=root_dir, transform=aug_transform)

    targets = np.array(base_dataset.targets)

    splitter = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=seed,
    )

    folds = list(splitter.split(np.zeros(len(targets)), targets))
    train_idx, val_idx = folds[fold_index]

    train_base = Subset(base_dataset, train_idx)

    if hyperparameters["augmentation"]:
        train_aug = Subset(aug_dataset, train_idx)
        train_data = ConcatDataset([train_base, train_aug])
    else:
        train_data = train_base

    val_data = Subset(base_dataset, val_idx)

    return train_data, val_data, len(base_dataset.classes), base_dataset.class_to_idx

# ----------------------------------------------
# Graph plotting function for training curves
# ----------------------------------------------

def plot_training_curves(history, runname, out_dir="modeldata"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["test_loss"], label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["test_acc"], label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(runname)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{runname}_curves.png")
    fig.savefig(out_path, dpi=150)
    
    from IPython.display import display
    display(fig)

    plt.close(fig)

    print(f"Saved training curves: {out_path}")
    return out_path


def plot_confusion_matrix(y_true, y_pred, class_names, runname, out_dir="modeldata"):
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True,
        values_format="d",
    )

    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{runname}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    
    from IPython.display import display
    display(fig)
    
    plt.close(fig)

    print(f"Saved confusion matrix: {out_path}")
    return out_path


# Plot ROC curve for multi-class classification
def plot_roc_curve(y_true, y_score, class_names, runname, out_dir="modeldata"):
    os.makedirs(out_dir, exist_ok=True)

    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.2f}")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{runname}_roc_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    from IPython.display import display
    display(fig)

    print(f"Saved ROC curve: {out_path}")
    return out_path

# ----------------------------------------------
# BitNetMCU training
# ----------------------------------------------

def create_run_name(hyperparameters):
    runname = (
        hyperparameters["runtag"]
        + "_" + hyperparameters["model"]
        + ("_Aug" if hyperparameters["augmentation"] else "")
        + "_BitMnist_" + hyperparameters["QuantType"]
        + "_width" + str(hyperparameters["network_width1"])
        + "_" + str(hyperparameters["network_width2"])
        + "_" + str(hyperparameters["network_width3"])
        + "_epochs" + str(hyperparameters["num_epochs"])
    )
    hyperparameters["runname"] = runname
    return runname


def load_model(model_name, params):
    try:
        module = importlib.import_module("models")
        model_class = getattr(module, model_name)

        kwargs = dict(
            network_width1=params["network_width1"],
            network_width2=params["network_width2"],
            network_width3=params["network_width3"],
            QuantType=params["QuantType"],
            NormType=params["NormType"],
            WScale=params["WScale"],
        )

        if "cnn_width" in params:
            kwargs["cnn_width"] = params["cnn_width"]

        if "num_classes" in params:
            kwargs["num_classes"] = params["num_classes"]

        return model_class(**kwargs)

    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py")


def add_mask_regularization(model, lambda_l1):
    mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

    if mask_layer is None:
        return 0

    return lambda_l1 * torch.norm(mask_layer.mask, 1)


def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
    total_activations = 0
    positive_activations = 0

    def hook_fn(module, input, output):
        nonlocal total_activations, positive_activations
        if isinstance(module, nn.ReLU) or isinstance(module, Activation):
            total_activations += output.numel()
            positive_activations += (output > 0).sum().item()

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_test_images), batch_size):
            images = all_test_images[i:i + batch_size]
            model(images)

    for hook in hooks:
        hook.remove()

    if total_activations == 0:
        return 0.0

    fraction_positive = positive_activations / total_activations
    writer.add_scalar("Activations/positive_fraction", fraction_positive, epoch + 1)

    return fraction_positive


def train_model(model, device, hyperparameters, train_data, test_data, class_names=None):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname = create_run_name(hyperparameters)

    batch_size = hyperparameters["batch_size"]
    
    seed = hyperparameters.get("seed", )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if hyperparameters["augmentation"]:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=loader_generator,
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_data, 
            batch_size=len(train_data), 
            shuffle=False,
            generator=loader_generator,
            worker_init_fn=seed_worker,
        )
        entire_dataset = next(iter(train_loader))
        all_train_images = entire_dataset[0].to(device)
        all_train_labels = entire_dataset[1].to(device)

    test_loader = DataLoader(
        test_data, 
        batch_size=len(test_data), 
        shuffle=False,
        generator = loader_generator,
        worker_init_fn=seed_worker
    )
    entire_dataset = next(iter(test_loader))
    all_test_images = entire_dataset[0].to(device)
    all_test_labels = entire_dataset[1].to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=hyperparameters.get("weight_decay", 0.0)
    )

    if hyperparameters["scheduler"] == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=hyperparameters["step_size"],
            gamma=hyperparameters["lr_decay"],
        )
    elif hyperparameters["scheduler"] == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    elif hyperparameters["scheduler"] == "CosineWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparameters["T_0"],
            T_mult=hyperparameters["T_mult"],
            eta_min=0,
        )
    else:
        raise ValueError("Invalid scheduler")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=hyperparameters.get("label_smoothing", 0.0)
    )

    os.makedirs("runs", exist_ok=True)
    os.makedirs("modeldata", exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{runname}-{now_str}")

    best_test_acc = 0.0
    best_test_loss = float("inf")
    best_state = None
    best_epoch_line = None
    totalbits = 0
    
    best_train_acc = 0.0
    best_train_loss = float("inf")
    
    best_y_true = None
    best_y_pred = None
    best_y_score = None # For ROC curve

    history ={
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        train_losses = []
        start_time = time.time()

        if hyperparameters["augmentation"]:
            total_train = len(train_loader.dataset)

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if epoch < hyperparameters.get("prune_epoch", -1):
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                correct += (predicted == labels).sum().item()

        else:
            total_train = len(all_train_images)
            indices = list(range(total_train))
            random.shuffle(indices)

            for i in range(0, total_train, batch_size):
                batch_indices = indices[i:i + batch_size]

                images = all_train_images[batch_indices]
                labels = all_train_labels[batch_indices]

                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if epoch < hyperparameters.get("prune_epoch", -1):
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                correct += (predicted == labels).sum().item()

        scheduler.step()

        if epoch + 1 == halve_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            print(f"Learning rate halved at epoch {epoch + 1}")

        trainaccuracy = correct / total_train * 100

        model.eval()
        correct = 0
        total = 0
        test_losses = []
        epoch_y_true = []
        epoch_y_pred = []
        epoch_y_score = []

        with torch.no_grad():
            for i in range(0, len(all_test_images), batch_size):
                images = all_test_images[i:i + batch_size] 
                labels = all_test_labels[i:i + batch_size]

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)

                
                epoch_y_true.extend(labels.cpu().numpy())
                epoch_y_pred.extend(predicted.cpu().numpy())
                    
                epoch_y_score.extend(probs.cpu().numpy())

                loss = criterion(outputs, labels)
                test_losses.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        testaccuracy = correct / total * 100
        mean_test_loss = np.mean(test_losses)

        mean_train_loss = np.mean(train_losses)

        if (best_state is None) or (trainaccuracy > best_train_acc) or (
            trainaccuracy == best_train_acc and mean_train_loss < best_train_loss
        ):
            
#        if (best_state is None) or (testaccuracy > best_test_acc) or (
#            testaccuracy == best_test_acc and mean_test_loss < best_test_loss
#        ):
        
            best_train_acc = trainaccuracy
            best_train_loss = mean_train_loss
            best_test_acc = testaccuracy
            best_test_loss = mean_test_loss

            best_state = model.state_dict()

            best_y_true = list(epoch_y_true)
            best_y_pred = list(epoch_y_pred)
            
            best_y_score = np.array(epoch_y_score)  # Store the best epoch's scores for ROC curve

            best_epoch_line = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"LTrain:{mean_train_loss:.6f} "
                f"ATrain:{trainaccuracy:.2f}% "
                f"LTest:{np.mean(test_losses):.6f} "
                f"ATest:{testaccuracy:.2f}% "
            )
        
        activity = log_positive_activations(model, writer, epoch, all_test_images, batch_size)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"LTrain:{np.mean(train_losses):.6f} "
            f"ATrain:{trainaccuracy:.2f}% "
            f"LTest:{np.mean(test_losses):.6f} "
            f"ATest:{testaccuracy:.2f}% "
            f"Time[s]:{epoch_time:.2f} "
            f"Act:{activity * 100:.1f}% "
            f"w_clip/entropy[bits]: ",
            end="",
        )

        totalbits = 0

        for layer in model.modules():
            if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

                if epoch < hyperparameters["maxw_update_until_epoch"]:
                    layer.update_clipping_scalar(
                        layer.weight,
                        hyperparameters["maxw_algo"],
                        hyperparameters["maxw_quantscale"],
                    )

                w_quant, _, _ = layer.weight_quant(layer.weight)
                _, counts = np.unique(w_quant.cpu().detach().numpy(), return_counts=True)
                probabilities = counts / np.sum(counts)
                entropy = -np.sum(probabilities * np.log2(probabilities))

                print(f"{layer.s.item():.3f}/{entropy:.2f}", end=" ")

                totalbits += layer.weight.numel() * layer.bpw

        print()

        if epoch + 1 == hyperparameters.get("prune_epoch", -1):
            for m in model.modules():
                if isinstance(m, MaskingLayer):
                    m.prune_channels(
                        prune_number=hyperparameters["prune_groupstoprune"],
                        groups=hyperparameters["prune_totalgroups"],
                    )

        writer.add_scalar("Loss/train", np.mean(train_losses), epoch + 1)
        writer.add_scalar("Accuracy/train", trainaccuracy, epoch + 1)
        writer.add_scalar("Loss/test", np.mean(test_losses), epoch + 1)
        writer.add_scalar("Accuracy/test", testaccuracy, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()
        
        history["train_loss"].append(np.mean(train_losses))
        history["test_loss"].append(np.mean(test_losses))
        history["train_acc"].append(trainaccuracy)
        history["test_acc"].append(testaccuracy)

    numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Best Epoch: {best_epoch_line}")
    print(f"TotalBits: {totalbits} TotalBytes: {totalbits / 8.0}")
    print(f"Best Validation Accuracy: {best_test_acc:.2f}%")


    writer.add_hparams(
        hyperparameters,
        {
            "Parameters": numofweights,
            "Totalbits": totalbits,
            "Accuracy/train": trainaccuracy,
            "Accuracy/test": testaccuracy,
            "Loss/train": np.mean(train_losses),
            "Loss/test": np.mean(test_losses),
        },
    )
    
    plot_training_curves(history, runname)
    
    if best_y_true is not None and best_y_pred is not None:
        if class_names is None:
            class_names = [str(i) for i in range(hyperparameters["num_classes"])]

        plot_confusion_matrix(best_y_true, best_y_pred, class_names, runname)
        plot_roc_curve(best_y_true, best_y_score, class_names, runname)

    writer.close()

    return best_state


def build_imagefolder_dataset(hyperparameters):
    train_dir = hyperparameters.get("train_folder", "training_set")
    test_dir = hyperparameters.get("validation_folder", "validation_set")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing folder not found: {test_dir}")

    mean = hyperparameters.get("mean", [0.1307])
    std = hyperparameters.get("std", [0.3081])

    # ensure tuple of floats
    if isinstance(mean, (float, int)):
        mean = (float(mean),)
    else:
        mean = tuple(mean)

    if isinstance(std, (float, int)):
        std = (float(std),)
    else:
        std = tuple(std)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.normalize(mean, std),
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    if train_data.classes != test_data.classes:
        raise ValueError(
            f"Train classes {train_data.classes} do not match test classes {test_data.classes}"
        )

    if hyperparameters["augmentation"]:
        aug_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
            transforms.RandomAffine(
                degrees=hyperparameters["rotation2"],
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
        ]

        if hyperparameters.get("horizontal_flip", False):
            aug_list.append(
                transforms.RandomHorizontalFlip(
                    p=hyperparameters.get("horizontal_flip_prob", 0.5)
                )
            )

        if hyperparameters.get("color_jitter", False):
            aug_list.append(
                transforms.ColorJitter(
                    brightness=hyperparameters.get("brightness", 0.2),
                    contrast=hyperparameters.get("contrast", 0.2),
                )
            )

        aug_list.extend([
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=40.0, sigma=4.0)
            ], p=hyperparameters["elastictransformprobability"]),
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.normalize(mean, std),
        ])

        if hyperparameters.get("random_erasing", False):
            aug_list.append(
                transforms.RandomErasing(
                    p=hyperparameters.get("random_erasing_prob", 0.1),
                    scale=(
                        hyperparameters.get("random_erasing_scale_min", 0.02), 
                        hyperparameters.get("random_erasing_scale_max", 0.08)
                    ),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            )

        augmented_transform = transforms.Compose(aug_list)

        augmented_train_data1 = datasets.ImageFolder(root=train_dir, transform=augmented_transform)
        train_data = ConcatDataset([
            train_data, 
            augmented_train_data1,
        ])

    class_to_idx = datasets.ImageFolder(root=train_dir).class_to_idx
    num_classes = len(class_to_idx)

    print("Dataset mode: IMAGEFOLDER")
    print(f"Train folder: {train_dir}")
    print(f"Test folder : {test_dir}")
    print(f"Classes     : {class_to_idx}")
    print(f"Num classes : {num_classes}")
    print(f"Train images: {len(train_data)}")
    print(f"Test images : {len(test_data)}")

    return train_data, test_data, num_classes, class_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--params", type=str, default="trainingparameters.yaml")
    args = parser.parse_args()

    paramname = args.params

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)
        
    seed = hyperparameters.get("seed", )
    deterministic = hyperparameters.get("deterministic", True)

    seed_everything(seed, deterministic)
    print(f"Seed: {seed}")

    runname = create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if hyperparameters.get("kfold", False):
        k_folds = hyperparameters.get("k_folds", 5)

        for fold_index in range(k_folds):
            print(f"\n===== Fold {fold_index + 1}/{k_folds} =====")

            fold_params = dict(hyperparameters)
            fold_params["runtag"] = f'{hyperparameters["runtag"]}_fold{fold_index + 1}'
            fold_params["seed"] = hyperparameters.get("seed", ) + fold_index

            train_data, test_data, num_classes, class_to_idx = build_imagefolder_kfold_datasets(
                fold_params,
                fold_index,
            )

            fold_params["num_classes"] = num_classes
            runname = create_run_name(fold_params)

            model = load_model(fold_params["model"], fold_params).to(device)
            
            class_names = [
                name for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1])
            ]

            print("training...")
            best_state = train_model(model, device, fold_params, train_data, test_data)

            os.makedirs("modeldata", exist_ok=True)
            torch.save(best_state, f"modeldata/{runname}.pth")
            print(f"Saved fold model: modeldata/{runname}.pth")

        sys.exit(0)

    dataset_name = hyperparameters.get("dataset", "IMAGEFOLDER").upper()

    if dataset_name == "IMAGEFOLDER":
        train_data, test_data, num_classes, class_to_idx = build_imagefolder_dataset(hyperparameters)

    elif dataset_name == "MNIST":
        num_classes = 10
        mean, std = (0.1307,), (0.3081,)

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.normalize(mean, std),
        ])

        train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)
        class_to_idx = {str(i): i for i in range(10)}

    elif dataset_name.startswith("EMNIST"):
        split = dataset_name.split("_")[1].lower() if "_" in dataset_name else "balanced"

        split_alias = {
            "BALANCED": "balanced",
            "BYCLASS": "byclass",
            "BYMERGE": "bymerge",
            "LETTERS": "letters",
            "DIGITS": "digits",
            "MNIST": "mnist",
        }

        split = split_alias.get(split.upper(), split)

        split_classes = {
            "byclass": 62,
            "bymerge": 47,
            "balanced": 47,
            "letters": 26,
            "digits": 10,
            "mnist": 10,
        }

        num_classes = split_classes.get(split, 26)
        mean, std = (0.1307,), (0.3081,)

        from torchvision.datasets import EMNIST

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.normalize(mean, std),
        ])

        train_data = EMNIST(root="data", split=split, train=True, transform=transform, download=True)
        test_data = EMNIST(root="data", split=split, train=False, transform=transform, download=True)
        class_to_idx = {str(i): i for i in range(num_classes)}

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    hyperparameters["num_classes"] = num_classes

    os.makedirs("modeldata", exist_ok=True)

    with open(f"modeldata/{runname}_class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)

    with open(f"modeldata/{runname}_used_params.yaml", "w") as f:
        yaml.dump(hyperparameters, f)

    model = load_model(hyperparameters["model"], hyperparameters)
    model = model.to(device)

    summary(model, input_size=(1, 16, 16))
    
    class_names = [
        name for name, idx in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    print("training...")
    best_state = train_model(model, device, hyperparameters, train_data, test_data, class_names=class_names)

    print("saving model...")

    if best_state is not None:
        torch.save(best_state, f"modeldata/{runname}.pth")
    else:
        torch.save(model.state_dict(), f"modeldata/{runname}.pth")

    print(f"Saved: modeldata/{runname}.pth")
    print(f"Saved: modeldata/{runname}_class_to_idx.json")
    
    manifest = {
        "runname": runname,
        "files": {
            "model": f"modeldata/{runname}.pth",
            "class_mapping": f"modeldata/{runname}_class_to_idx.json",
            "used_params": f"modeldata/{runname}_used_params.yaml",
        },

        "dataset": {
            "dataset": hyperparameters.get("dataset"),
            "train_folder": hyperparameters.get("train_folder"),
            "validation_folder": hyperparameters.get("validation_folder"),
            "test_folder": hyperparameters.get("test_folder"),
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "mean": hyperparameters.get("mean"),
            "std": hyperparameters.get("std"),
        },

        "model": {
            "model": hyperparameters.get("model"),
            "runtag": hyperparameters.get("runtag"),
            "QuantType": hyperparameters.get("QuantType"),
            "NormType": hyperparameters.get("NormType"),
            "WScale": hyperparameters.get("WScale"),
            "network_width1": hyperparameters.get("network_width1"),
            "network_width2": hyperparameters.get("network_width2"),
            "network_width3": hyperparameters.get("network_width3"),
            "cnn_width": hyperparameters.get("cnn_width"),
        },

        "training": {
            "num_epochs": hyperparameters.get("num_epochs"),
            "batch_size": hyperparameters.get("batch_size"),
            "learning_rate": hyperparameters.get("learning_rate"),
            "label_smoothing": hyperparameters.get("label_smoothing", 0.0),
            "weight_decay": hyperparameters.get("weight_decay", 0.0),
            "best_loss_min_delta": hyperparameters.get("best_loss_min_delta"),
            "seed": hyperparameters.get("seed"),
            "deterministic": hyperparameters.get("deterministic"),
        },

        "scheduler": {
            "scheduler": hyperparameters.get("scheduler"),
            "step_size": hyperparameters.get("step_size"),
            "lr_decay": hyperparameters.get("lr_decay"),
            "T_0": hyperparameters.get("T_0"),
            "T_mult": hyperparameters.get("T_mult"),
        },

        "augmentation": {
            "augmentation": hyperparameters.get("augmentation"),
            "rotation1": hyperparameters.get("rotation1"),
            "rotation2": hyperparameters.get("rotation2"),
            "elastictransformprobability": hyperparameters.get("elastictransformprobability"),
            "horizontal_flip": hyperparameters.get("horizontal_flip"),
            "horizontal_flip_prob": hyperparameters.get("horizontal_flip_prob"),
            "color_jitter": hyperparameters.get("color_jitter"),
            "brightness": hyperparameters.get("brightness"),
            "contrast": hyperparameters.get("contrast"),
            "random_erasing": hyperparameters.get("random_erasing"),
            "random_erasing_prob": hyperparameters.get("random_erasing_prob"),
            "random_erasing_scale_min": hyperparameters.get("random_erasing_scale_min"),
            "random_erasing_scale_max": hyperparameters.get("random_erasing_scale_max"),
        },

        "regularization_pruning": {
            "lambda_l1": hyperparameters.get("lambda_l1"),
            "prune_epoch": hyperparameters.get("prune_epoch"),
            "prune_groupstoprune": hyperparameters.get("prune_groupstoprune"),
            "prune_totalgroups": hyperparameters.get("prune_totalgroups"),
        },

        "maxw": {
            "maxw_algo": hyperparameters.get("maxw_algo"),
            "maxw_update_until_epoch": hyperparameters.get("maxw_update_until_epoch"),
            "maxw_quantscale": hyperparameters.get("maxw_quantscale"),
        },

        "kfold": {
            "kfold": hyperparameters.get("kfold"),
            "k_folds": hyperparameters.get("k_folds"),
            "kfold_folder": hyperparameters.get("kfold_folder"),
        },
    }

    with open(f"modeldata/{runname}_manifest.yaml", "w") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print(f"Saved: modeldata/{runname}_manifest.yaml")
