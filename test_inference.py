# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import numpy as np
# from BitNetMCU import QuantizedModel
# # from models import FCMNIST
# from ctypes import CDLL, c_uint32, c_int8, c_uint8, POINTER
# import argparse
# import yaml
# import importlib

# # Export quantized model from saved checkpoint
# # cpldcpu 2024-04-14
# # Note: Hyperparameters are used to generated the filename
# #---------------------------------------------

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
    
# def export_test_data_to_c(test_loader, filename, num=8):
#     with open(filename, 'w') as f:
#         for i, (input_data, labels) in enumerate(test_loader):
#             if i >= num:
#                 break
#             # Reshape and convert to numpy
#             input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#             labels = labels.cpu().numpy()

#             scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#             scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.uint8)

#             f.write(f'int8_t input_data_{i}[256] = {{\n')
#             flattened_data = scaled_data.flatten()
#             for k in range(0, len(flattened_data), 16):
#                 f.write(', '.join(f'0x{value:02X}' for value in flattened_data[k:k+16]) + ',\n')
#             f.write('};\n')

#             f.write(f'uint8_t label_{i} = ' + str(labels[0]) + ';\n')

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

#     # main
#     runname= create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the MNIST dataset
#     transform = transforms.Compose([
#         transforms.Resize((16, 16)),  # Resize images to 16x16
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     #train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
#     #test_data = datasets.MNIST(root='data', train=False, transform=transform)
#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     #if dataset_name == "MNIST":
#         #test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

#     #elif dataset_name == "EMNIST_DIGITS":
#          #test_data = datasets.EMNIST(root='data', split='digits', train=False, transform=transform, download=True)

#     #else:
#          #raise ValueError(f"Unsupported dataset: {dataset_name}")
#     if dataset_name == "MNIST":
#         num_classes = 10
#         test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

#     elif dataset_name == "EMNIST_DIGITS":
#           num_classes = 10
#           test_data = datasets.EMNIST(root='data', split='digits', train=False, transform=transform, download=True)

#     elif dataset_name == "EMNIST_LETTERS":
#           num_classes = 26
#           test_data = datasets.EMNIST(root='data', split='letters', train=False, transform=transform, download=True)

#     else:
#          raise ValueError(f"Unsupported dataset: {dataset_name}")

#     hyperparameters["num_classes"] = num_classes
    
#     # Create data loaders
#     test_loader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

#     model = load_model(hyperparameters["model"], hyperparameters).to(device)
    
#     print('Loading model...')    
#     try:
#         model.load_state_dict(torch.load(f'modeldata/{runname}.pth'))
#     except FileNotFoundError:
#         print(f"The file 'modeldata/{runname}.pth' does not exist.")
#         exit()

#     print('Inference using the original model...')
#     correct = 0
#     total = 0
#     test_loss = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)        
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     testaccuracy = correct / total * 100
#     print(f'Accuracy/Test of trained model: {testaccuracy} %')

#     print('Quantizing model...')
#     # Quantize the model
#     quantized_model = QuantizedModel(model)
#     print(f'Total number of bits: {quantized_model.totalbits()} ({quantized_model.totalbits()/8/1024} kbytes)')

#     # Inference using the quantized model
#     print ("Verifying inference of quantized model in Python and C")

#    # Initialize counter
#     counter = 0
#     correct_c = 0
#     correct_py = 0
#     mismatch = 0

#     test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)    

#     # export_test_data_to_c(test_loader2, 'BitNetMCU_MNIST_test_data.h', num=10)

#     lib = CDLL('./Bitnet_inf.dll')

#     for input_data, labels in test_loader2:
#         input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#         labels = labels.cpu().numpy()

#         scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#         scaled_data = np.round(input_data * scale).clip(-128, 127) 

#         # Create a pointer to the ctypes array
#         input_data_pointer = (c_int8 * len(scaled_data.flatten()))(*scaled_data.astype(np.int8).flatten())

#         lib.Inference.argtypes = [POINTER(c_int8)]
#         lib.Inference.restype = c_uint32

#         # Inference C
#         result_c = lib.Inference(input_data_pointer)

#         # Inference Python
#         result_py = quantized_model.inference_quantized(input_data)
#         predict_py = np.argmax(result_py, axis=1)

#         # activations = quantized_model.get_activations(input_data)

#         if (result_c == labels[0]):
#             correct_c += 1

#         if (predict_py[0] == labels[0]):
#             correct_py += 1

#         if (result_c != predict_py[0]):
#             print(f'{counter:5} Mismatch between inference engines found. Prediction C: {result_c} Prediction Python: {predict_py[0]} True: {labels[0]}')
#             mismatch +=1

#         counter += 1

#     print("size of test data:", counter)
#     print(f'Mispredictions C: {counter - correct_c} Py: {counter - correct_py}')
#     print('Overall accuracy C:', correct_c / counter * 100, '%')
#     print('Overall accuracy Python:', correct_py / counter * 100, '%')
    
#     print(f'Mismatches between engines: {mismatch} ({mismatch/counter*100}%)')

import os
import json
import csv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from BitNetMCU import QuantizedModel
from ctypes import CDLL, c_uint32, c_int8, POINTER
import argparse
import yaml
import importlib


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


def load_class_mapping(runname):
    mapping_file = f"modeldata/{runname}_class_to_idx.json"

    if not os.path.exists(mapping_file):
        return None, None

    with open(mapping_file, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return class_to_idx, idx_to_class


def get_transform(hyperparameters):
    mean = hyperparameters.get("mean", [0.4360])
    std = hyperparameters.get("std", [0.1722])

    if isinstance(mean, (float, int)):
        mean = (float(mean),)
    else:
        mean = tuple(mean)

    if isinstance(std, (float, int)):
        std = (float(std),)
    else:
        std = tuple(std)

    return transforms.Compose([
        transforms.grayscale(num_output_channels=1),  # Ensure single channel for MNIST/EMNIST
        transforms.Resize((16, 16)),  # adjust if needed
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_test_dataset(hyperparameters):
    dataset_name = hyperparameters.get("dataset", "IMAGEFOLDER").upper()
    transform = get_transform(hyperparameters)

    if dataset_name == "IMAGEFOLDER":
        test_dir = hyperparameters.get("test_folder", "testing_set")

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Testing folder not found: {test_dir}")

        test_data = datasets.ImageFolder(root=test_dir, transform=transform)
        num_classes = len(test_data.classes)

        print("Dataset mode: IMAGEFOLDER")
        print(f"Test folder : {test_dir}")
        print(f"Classes     : {test_data.class_to_idx}")
        print(f"Num classes : {num_classes}")
        print(f"Test images : {len(test_data)}")

        return test_data, num_classes, test_data.class_to_idx

    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.4360,), (0.1722,)),
        ])

        test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)
        class_to_idx = {str(i): i for i in range(10)}
        return test_data, 10, class_to_idx

    elif dataset_name == "EMNIST_DIGITS":
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        test_data = datasets.EMNIST(
            root="data",
            split="digits",
            train=False,
            transform=transform,
            download=True,
        )

        class_to_idx = {str(i): i for i in range(10)}
        return test_data, 10, class_to_idx

    elif dataset_name == "EMNIST_LETTERS":
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        test_data = datasets.EMNIST(
            root="data",
            split="letters",
            train=False,
            transform=transform,
            download=True,
        )

        class_to_idx = {str(i): i for i in range(26)}
        return test_data, 26, class_to_idx

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def export_test_data_to_c(test_loader, filename, num=8):
    with open(filename, "w") as f:
        for i, (input_data, labels) in enumerate(test_loader):
            if i >= num:
                break

            input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
            labels = labels.cpu().numpy()

            scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
            scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.int8)

            f.write(f"int8_t input_data_{i}[256] = {{\n")

            flattened_data = scaled_data.flatten()
            for k in range(0, len(flattened_data), 16):
                f.write(", ".join(f"{value}" for value in flattened_data[k:k + 16]) + ",\n")

            f.write("};\n")
            f.write(f"uint8_t label_{i} = {int(labels[0])};\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--params", type=str, default="trainingparameters.yaml")
    parser.add_argument("--skip-c", action="store_true", help="Skip C DLL inference check")
    parser.add_argument("--export-c-test", action="store_true", help="Export test samples to C header")
    args = parser.parse_args()

    paramname = args.params

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    runname = create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_data, num_classes, dataset_class_to_idx = build_test_dataset(hyperparameters)
    hyperparameters["num_classes"] = num_classes

    saved_class_to_idx, saved_idx_to_class = load_class_mapping(runname)

    if saved_class_to_idx is not None:
        print(f"Saved class mapping: {saved_class_to_idx}")

        if saved_class_to_idx != dataset_class_to_idx:
            raise ValueError(
                "Class mapping mismatch between training and testing.\n"
                f"Saved mapping: {saved_class_to_idx}\n"
                f"Test mapping : {dataset_class_to_idx}"
            )

        idx_to_class = saved_idx_to_class
    else:
        print("Warning: class_to_idx file not found. Using dataset folder class order.")
        idx_to_class = {v: k for k, v in dataset_class_to_idx.items()}

    test_loader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
    )

    model = load_model(hyperparameters["model"], hyperparameters).to(device)

    print("Loading model...")

    model_path = f"modeldata/{runname}.pth"

    if not os.path.exists(model_path):
        print(f"The file '{model_path}' does not exist.")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Inference using the original model...")

    correct = 0
    total = 0
    prediction_rows = []

    with torch.no_grad():
        sample_index = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for b in range(labels.size(0)):
                true_idx = int(labels[b].cpu().item())
                pred_idx = int(predicted[b].cpu().item())
                conf = float(confidence[b].cpu().item())

                prediction_rows.append({
                    "sample_index": sample_index,
                    "true_idx": true_idx,
                    "true_class": idx_to_class[true_idx],
                    "pred_idx": pred_idx,
                    "pred_class": idx_to_class[pred_idx],
                    "confidence": conf,
                    "correct": true_idx == pred_idx,
                })

                sample_index += 1

    testaccuracy = correct / total * 100
    print(f"Accuracy/Test of trained model: {testaccuracy:.2f} %")

    os.makedirs("prediction_results", exist_ok=True)
    csv_path = f"prediction_results/{runname}_predictions.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "true_idx",
                "true_class",
                "pred_idx",
                "pred_class",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Saved predictions: {csv_path}")

    print("\nFirst predictions:")
    for row in prediction_rows[:20]:
        print(
            f"{row['sample_index']:03d} "
            f"True:{row['true_class']} "
            f"Pred:{row['pred_class']} "
            f"Conf:{row['confidence'] * 100:.1f}% "
            f"{'OK' if row['correct'] else 'WRONG'}"
        )

    if args.export_c_test:
        test_loader_single = DataLoader(test_data, batch_size=1, shuffle=False)
        export_test_data_to_c(test_loader_single, f"{runname}_test_data.h", num=10)
        print(f"Exported C test data: {runname}_test_data.h")

    print("\nQuantizing model...")
    quantized_model = QuantizedModel(model)
    print(
        f"Total number of bits: {quantized_model.totalbits()} "
        f"({quantized_model.totalbits() / 8 / 1024:.2f} kbytes)"
    )

    print("Verifying inference of quantized model in Python")

    counter = 0
    correct_py = 0

    test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

    for input_data, labels in test_loader2:
        input_flat = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        result_py = quantized_model.inference_quantized(input_flat)
        predict_py = np.argmax(result_py, axis=1)

        if predict_py[0] == labels_np[0]:
            correct_py += 1

        counter += 1

    print("size of test data:", counter)
    print(f"Mispredictions Python quantized: {counter - correct_py}")
    print("Overall accuracy Python quantized:", correct_py / counter * 100, "%")

    if args.skip_c:
        print("Skipped C DLL inference check.")
        exit()

    if not os.path.exists("./Bitnet_inf.dll"):
        print("Bitnet_inf.dll not found. Skipping C inference check.")
        exit()

    print("\nVerifying inference of quantized model in C DLL")

    lib = CDLL("./Bitnet_inf.dll")
    lib.Inference.argtypes = [POINTER(c_int8)]
    lib.Inference.restype = c_uint32

    counter = 0
    correct_c = 0
    correct_py = 0
    mismatch = 0

    test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

    for input_data, labels in test_loader2:
        input_flat = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        scale = 127.0 / np.maximum(np.abs(input_flat).max(axis=-1, keepdims=True), 1e-5)
        scaled_data = np.round(input_flat * scale).clip(-128, 127)

        input_data_pointer = (
            c_int8 * len(scaled_data.flatten())
        )(*scaled_data.astype(np.int8).flatten())

        result_c = lib.Inference(input_data_pointer)

        result_py = quantized_model.inference_quantized(input_flat)
        predict_py = np.argmax(result_py, axis=1)

        if result_c == labels_np[0]:
            correct_c += 1

        if predict_py[0] == labels_np[0]:
            correct_py += 1

        if result_c != predict_py[0]:
            print(
                f"{counter:5} Mismatch. "
                f"Prediction C: {result_c} "
                f"Prediction Python: {predict_py[0]} "
                f"True: {labels_np[0]}"
            )
            mismatch += 1

        counter += 1

    print("size of test data:", counter)
    print(f"Mispredictions C: {counter - correct_c} Py: {counter - correct_py}")
    print("Overall accuracy C:", correct_c / counter * 100, "%")
    print("Overall accuracy Python:", correct_py / counter * 100, "%")
    print(f"Mismatches between engines: {mismatch} ({mismatch / counter * 100}%)")
