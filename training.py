# import torch, torch.nn as nn, torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import ConcatDataset
# from datetime import datetime
# # from models import FCMNIST, CNNMNIST
# from BitNetMCU import BitLinear, BitConv2d, Activation
# import time
# import random
# import argparse
# import yaml
# from torchsummary import summary
# import importlib
# from models import MaskingLayer

# #----------------------------------------------
# # BitNetMCU training
# #----------------------------------------------

# def create_run_name(hyperparameters):
#     runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
#     hyperparameters["runname"] = runname
#     return runname

# def load_model(model_name, params):
#     try:
#         module = importlib.import_module('models')
#         model_class = getattr(module, model_name)
#         kwargs = dict(
#             network_width1=params["network_width1"],
#             network_width2=params["network_width2"],
#             network_width3=params["network_width3"],
#             QuantType=params["QuantType"],
#             NormType=params["NormType"],
#             WScale=params["WScale"]
#         )
#         if 'cnn_width' in params:
#             kwargs['cnn_width'] = params['cnn_width']
#         if 'num_classes' in params:
#             kwargs['num_classes'] = params['num_classes']
#         return model_class(**kwargs)
#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py")

# def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
#     total_activations = 0
#     positive_activations = 0

#     def hook_fn(module, input, output):
#         nonlocal total_activations, positive_activations
#         if isinstance(module, nn.ReLU) or isinstance(module, Activation):
#             total_activations += output.numel()
#             positive_activations += (output > 0).sum().item()

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     # Run a forward pass to trigger hooks
#     with torch.no_grad():
#         for i in range(len(all_test_images) // batch_size):
#             images = all_test_images[i * batch_size:(i + 1) * batch_size]
#             model(images)

#     for hook in hooks:
#         hook.remove()

#     fraction_positive = positive_activations / total_activations
#     writer.add_scalar('Activations/positive_fraction', fraction_positive, epoch+1)

#     return fraction_positive


# # Function to add L1 regularization on the mask
# def add_mask_regularization(model,  lambda_l1):
#     mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

#     if mask_layer is None:
#         return 0
    
#     l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
#     return l1_reg


# def train_model(model, device, hyperparameters, train_data, test_data):
#     num_epochs = hyperparameters["num_epochs"]
#     learning_rate = hyperparameters["learning_rate"]
#     halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
#     runname =  create_run_name(hyperparameters)

#     # define dataloaders

#     batch_size = hyperparameters["batch_size"]  # Define your batch size

#     # ON-the-fly augmentation requires using the (slow) dataloader. Without augmentation, we can load the entire dataset into GPU for speedup
#     if hyperparameters["augmentation"]:
#         train_loader = DataLoader(
#         train_data, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True)
#     else:
#         # load entire dataset into GPU for 5x speedup
#         train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False) # shuffling will be done separately
#         entire_dataset = next(iter(train_loader))
#         all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     # Test dataset is always in GPU
#     test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
#     entire_dataset = next(iter(test_loader))
#     all_test_images, all_test_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     if hyperparameters["scheduler"] == "StepLR":
#         scheduler = StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["lr_decay"])
#     elif hyperparameters["scheduler"] == "Cosine":
#         scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    
#     elif hyperparameters["scheduler"] == "CosineWarmRestarts":
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=hyperparameters["T_0"], T_mult=hyperparameters["T_mult"], eta_min=0)
#     else:
#         raise ValueError("Invalid scheduler")

#     criterion = nn.CrossEntropyLoss()

#     # tensorboard writer
#     now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir=f'runs/{runname}-{now_str}')

#     train_loss=[]
#     test_loss = []

#     # Train the CNN
#     for epoch in range(num_epochs):
#         correct = 0
#         train_loss=[]
#         start_time = time.time()

#         if hyperparameters["augmentation"]:
#             for i, (images, labels) in enumerate(train_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()
#         else:
#             # Shuffle images (important!)
#             indices = list(range(len(all_train_images)))
#             random.shuffle(indices)

#             for i in range(len(indices) // batch_size):
#                 batch_indices = indices[i * batch_size:(i + 1) * batch_size]
#                 images = torch.stack([all_train_images[i] for i in batch_indices])
#                 labels = torch.stack([all_train_labels[i] for i in batch_indices])
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         scheduler.step()

#         if epoch + 1 == halve_lr_epoch:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= 0.5
#             print(f"Learning rate halved at epoch {epoch + 1}")


#         trainaccuracy = correct / len(train_loader.dataset) * 100

#         correct = 0
#         total = 0
#         test_loss = []
#         with torch.no_grad():
#             for i in range(len(all_test_images) // batch_size):
#                 images = all_test_images[i * batch_size:(i + 1) * batch_size]
#                 labels = all_test_labels[i * batch_size:(i + 1) * batch_size]

#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 test_loss.append(loss.item())
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         # Log positive activations
#         activity=log_positive_activations(model, writer, epoch, all_test_images, batch_size)

#         end_time = time.time()
#         epoch_time = end_time - start_time

#         testaccuracy = correct / total * 100

#         print(f'Epoch [{epoch+1}/{num_epochs}], LTrain:{np.mean(train_loss):.6f} ATrain: {trainaccuracy:.2f}% LTest:{np.mean(test_loss):.6f} ATest: {correct / total * 100:.2f}% Time[s]: {epoch_time:.2f} Act: {activity*100:.1f}% w_clip/entropy[bits]: ', end='')

#         # update clipping scalars once per epoch
#         totalbits = 0
#         for i, layer in enumerate(model.modules()):
#             if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

#                 # update clipping scalar
#                 if epoch < hyperparameters['maxw_update_until_epoch']:
#                     layer.update_clipping_scalar(layer.weight, hyperparameters['maxw_algo'], hyperparameters['maxw_quantscale'])

#                 # calculate entropy of weights
#                 w_quant, _, _ = layer.weight_quant(layer.weight)
#                 _, counts = np.unique(w_quant.cpu().detach().numpy(), return_counts=True)
#                 probabilities = counts / np.sum(counts)
#                 entropy = -np.sum(probabilities * np.log2(probabilities))

#                 print(f'{layer.s.item():.3f}/{entropy:.2f}', end=' ')

#                 totalbits += layer.weight.numel() * layer.bpw

#         print()

#         if epoch + 1 == hyperparameters ["prune_epoch"]:
#             for m in model.modules():
#                 if isinstance(m, MaskingLayer):            
#                     pruned_channels, remaining_channels = m.prune_channels(prune_number=hyperparameters['prune_groupstoprune'], groups=hyperparameters['prune_totalgroups'])

#         writer.add_scalar('Loss/train', np.mean(train_loss), epoch+1)
#         writer.add_scalar('Accuracy/train', trainaccuracy, epoch+1)
#         writer.add_scalar('Loss/test', np.mean(test_loss), epoch+1)
#         writer.add_scalar('Accuracy/test', testaccuracy, epoch+1)
#         writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
#         writer.flush()

#     numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # totalbits = numofweights * hyperparameters['BPW']

#     print(f'TotalBits: {totalbits} TotalBytes: {totalbits/8.0} ')

#     writer.add_hparams(hyperparameters, {'Parameters': numofweights, 'Totalbits': totalbits, 'Accuracy/train': trainaccuracy, 'Accuracy/test': testaccuracy, 'Loss/train': np.mean(train_loss), 'Loss/test': np.mean(test_loss)})
#     writer.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Training script')
#     parser.add_argument('--params', type=str, help='Name of the parameter file', default='trainingparameters.yaml')

#     args = parser.parse_args()

#     if args.params:
#         paramname = args.params
#     else:
#         paramname = 'trainingparameters.yaml'

#     print(f'Load parameters from file: {paramname}')
#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     runname= create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset selection (MNIST default, EMNIST optional)
#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     if dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         base_dataset_train = datasets.MNIST
#         base_dataset_test = datasets.MNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}
#     elif dataset_name.startswith("EMNIST"):
#         # Expected format: EMNIST or EMNIST_BALANCED, EMNIST_BYCLASS etc.
#         # Torchvision subsets: 'byclass'(62), 'bymerge'(47), 'balanced'(47), 'letters'(26), 'digits'(10), 'mnist'(10)
#         split = dataset_name.split('_')[1].lower() if '_' in dataset_name else 'balanced'
#         # Map common names
#         split_alias = { 'BALANCED':'balanced', 'BYCLASS':'byclass', 'BYMERGE':'bymerge', 'LETTERS':'letters', 'DIGITS':'digits', 'MNIST':'mnist'}
#         split = split_alias.get(split.upper(), split)
#         # class counts per split
#         split_classes = { 'byclass':62, 'bymerge':47, 'balanced':47, 'letters':26, 'digits':10, 'mnist':10 }
#         num_classes = split_classes.get(split, 26)
#         # EMNIST uses same normalization as MNIST typically
#         mean, std = (0.1307,), (0.3081,)
#         from torchvision.datasets import EMNIST
#         base_dataset_train = EMNIST
#         base_dataset_test = EMNIST
#         dataset_kwargs = {"split": split, "train": True}
#         dataset_kwargs_test = {"split": split, "train": False}
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     transform = transforms.Compose([
#         transforms.Resize((16, 16)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])

#     train_data = base_dataset_train(root='data', transform=transform, download=True, **dataset_kwargs)
#     test_data = base_dataset_test(root='data', transform=transform, download=True, **dataset_kwargs_test)

#     if hyperparameters["augmentation"]:
#         # Data augmentation for training data
#         augmented_transform = transforms.Compose([
#             transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
#             transforms.RandomAffine(degrees=hyperparameters["rotation2"], translate=(0.1, 0.1), scale=(0.9, 1.1)),
#             transforms.RandomApply([
#                 transforms.ElasticTransform(alpha=40.0, sigma=4.0)
#             ], p=hyperparameters["elastictransformprobability"]),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         augmented_train_data = base_dataset_train(root='data', transform=augmented_transform, download=True, **dataset_kwargs)
#         train_data = ConcatDataset([train_data, augmented_train_data])

#     # Pass num_classes dynamically to model
#     hyperparameters['num_classes'] = num_classes
#     model = load_model(hyperparameters["model"], {**hyperparameters, 'num_classes': num_classes})
#     # If model class supports num_classes argument, it will be used. Otherwise ignore.
#     if hasattr(model, 'to'):
#         model = model.to(device)

#     summary(model, input_size=(1, 16, 16))  # Assuming the input size is (1, 16, 16)

#     print('training...')
#     train_model(model, device, hyperparameters, train_data, test_data)

#     print('saving model...')
#     torch.save(model.state_dict(), f'modeldata/{runname}.pth')


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


def train_model(model, device, hyperparameters, train_data, test_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname = create_run_name(hyperparameters)

    batch_size = hyperparameters["batch_size"]

    if hyperparameters["augmentation"]:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
        entire_dataset = next(iter(train_loader))
        all_train_images = entire_dataset[0].to(device)
        all_train_labels = entire_dataset[1].to(device)

    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    entire_dataset = next(iter(test_loader))
    all_test_images = entire_dataset[0].to(device)
    all_test_labels = entire_dataset[1].to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    criterion = nn.CrossEntropyLoss()

    os.makedirs("runs", exist_ok=True)
    os.makedirs("modeldata", exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{runname}-{now_str}")

    best_test_acc = 0.0
    best_test_loss = float("inf")
    best_state = None
    best_epoch_line = None
    totalbits = 0
    
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

        with torch.no_grad():
            for i in range(0, len(all_test_images), batch_size):
                images = all_test_images[i:i + batch_size]
                labels = all_test_labels[i:i + batch_size]

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                test_losses.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        testaccuracy = correct / total * 100
        mean_test_loss = np.mean(test_losses)

        if mean_test_loss < best_test_loss:
            best_test_acc = testaccuracy
            best_test_loss = mean_test_loss
            best_state = model.state_dict()
            best_epoch_line = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"LTrain:{np.mean(train_losses):.6f} "
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

    print(f"Best Validation Accuracy: {best_test_acc:.2f}%")
    print(f"Best Epoch: {best_epoch_line}")
    print(f"TotalBits: {totalbits} TotalBytes: {totalbits / 8.0}")

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

    writer.close()

    return best_state


def build_imagefolder_dataset(hyperparameters):
    train_dir = hyperparameters.get("train_folder", "training_set")
    test_dir = hyperparameters.get("validation_folder", "validation_set")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing folder not found: {test_dir}")

    mean = hyperparameters.get("mean", [0.4333])
    std = hyperparameters.get("std", [0.1472])

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
        transforms.Normalize(mean, std),
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    if train_data.classes != test_data.classes:
        raise ValueError(
            f"Train classes {train_data.classes} do not match test classes {test_data.classes}"
        )

    if hyperparameters["augmentation"]:
        augmented_transform = transforms.Compose([
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
            transforms.Normalize(mean, std),
        ])

        augmented_train_data = datasets.ImageFolder(root=train_dir, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

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

    runname = create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_name = hyperparameters.get("dataset", "IMAGEFOLDER").upper()

    if dataset_name == "IMAGEFOLDER":
        train_data, test_data, num_classes, class_to_idx = build_imagefolder_dataset(hyperparameters)

    elif dataset_name == "MNIST":
        num_classes = 10
        mean, std = (0.4333,), (0.1472,)

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
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
            transforms.Normalize(mean, std),
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
    
    # Load pretrained weights if specified
    pretrained_path = hyperparameters.get("pretrained_model", "")

    if pretrained_path:
        print(f"Loading pretrained weights: {pretrained_path}")

        pretrained_state = torch.load(pretrained_path, map_location=device)
        model_state = model.state_dict()

        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        skipped_state = [
            k for k, v in pretrained_state.items()
            if k not in model_state or v.shape != model_state[k].shape
        ]

        model_state.update(filtered_state)
        model.load_state_dict(model_state)

        print(f"Loaded pretrained tensors: {len(filtered_state)}")
        print(f"Skipped tensors: {skipped_state}")

    summary(model, input_size=(1, 16, 16))

    print("training...")
    best_state = train_model(model, device, hyperparameters, train_data, test_data)

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
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
        },
        "model": {
            "name": hyperparameters["model"],
            "quant_type": hyperparameters["QuantType"],
            "norm_type": hyperparameters["NormType"],
            "weight_scale": hyperparameters["WScale"],
            "network_width1": hyperparameters["network_width1"],
            "network_width2": hyperparameters["network_width2"],
            "network_width3": hyperparameters["network_width3"],
        },
        "training": {
            "num_epochs": hyperparameters["num_epochs"],
            "batch_size": hyperparameters["batch_size"],
            "learning_rate": hyperparameters["learning_rate"],
            "scheduler": hyperparameters["scheduler"],
            "augmentation": hyperparameters["augmentation"],
        },
    }

    with open(f"modeldata/{runname}_manifest.yaml", "w") as f:
        yaml.dump(manifest, f, sort_keys=False)

    print(f"Saved: modeldata/{runname}_manifest.yaml")
