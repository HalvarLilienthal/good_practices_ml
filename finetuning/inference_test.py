
import argparse
import sys

sys.path.append('.')
sys.path.append("../")
sys.path.append('./scripts')

from codecarbon import OfflineEmissionsTracker
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
import tqdm

from model import nn
from model.region_loss import Regional_Loss
from utils import load_dataset
import torch.nn.utils.prune as prune
import torch_tensorrt

def prune_model(model, prune_config: dict):
    for _, module in list(model.named_modules())[1:]:
        pruned = False
        if prune_config["structured"]:
            # prune 20% of connections in all conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_config["structured_conv"])
                pruned = True
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_config["structured_linear"])
                pruned = True
        else:
            prune.random_unstructured(module, name="weight", amount=prune_config["unstructured_all"])
            pruned = True
        if pruned:
            # Set the actual weight values to 0
            prune.remove(module, 'weight')
            # TODO: Use sparsity for optimization
            # module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
    return model

def quantise_model(model, quantise_config):
    return torch.compile(model, **quantise_config)

if __name__ == "__main__":
    """Runs the initial CLIP experiments
    """
    parser = argparse.ArgumentParser(description='Test inference time of finetuned model')
    parser.add_argument('--model_path', metavar='str', required=True,
                        help='The path to the finetuned model file')
    parser.add_argument('--yaml_path', metavar='str', required=False,
                        help='The path to the yaml file with the stored paths', default='paths.yaml')
    parser.add_argument('--batch_size', required=False,
                        help='The batch size', default=1)
    parser.add_argument('--iterations', required=False,
                        help='The number of iterations', default=100)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tests = ["default", "inference_pruning", "quantisation"]

    inference_pruning_config = {
        # Whether to use structured l1 weight pruning or unstructured random pruning
        "structured": True,
        # The percentage of weights to prune in Conv2d layers in structured pruning
        "structured_conv": 0.2,
        # The percentage of weights to prune in Linear layers in structured pruning
        "structured_linear": 0.4,
        # The percentage of weights to prune in All layers in unstructured pruning
        "unstructured_all": 0.3
    }
    quantise_config = {
        "backend": "torch_tensorrt"
        # See torch.compile for more parameters
    }
    
    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path']
        REPO_PATH = paths['repo_path']
        testing_directory = f'{REPO_PATH}/CLIP_Embeddings/Testing'

        test_df = pd.read_csv(f'{testing_directory}/known_test_data.csv')
        test_dataset = load_dataset.EmbeddingDataset_from_df(test_df, "test")
        test_loader = DataLoader(test_dataset, shuffle=False)

        country_list = f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv'
        country_list = pd.read_csv(country_list)
        criterion = Regional_Loss(country_list)

        for test in tests:
            # Load the model
            model = nn.FinetunedClip()
            model.load_state_dict(torch.load(args.model_path), strict=True)
            model.to(device)

            if test == "inference_pruning":
                model = prune_model(model, inference_pruning_config)
            elif test == "quantisation":
                model = quantise_model(model, quantise_config)

            # Initialize accumulators for accuracy
            total_test_region_accuracy = 0.0
            total_test_accuracy = 0.0
            total_samples = 0

            tracker = OfflineEmissionsTracker(
                experiment_id=f"inference_test_{test}",
                country_iso_code="DEU",
                measure_power_secs=5,
                project_name="inference_test.py",
                tracking_mode="process",
                log_level="error",
                output_dir=".",
                output_file="emissions.csv",
                allow_multiple_runs=True    # Set this to True to allow multiple instances of codecarbon to run at the same time
            )

            tracker.start()

            try:
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    for iteration in tqdm.tqdm(range(args.iterations)):
                        for batch_idx, (inputs, targets) in enumerate(test_loader):
                            # Move inputs and targets to the appropriate device
                            inputs = inputs.to(device)
                            
                            # Perform forward pass
                            outputs = model(inputs)
                            
                            # Accumulate batch accuracies
                            batch_region_accuracy = criterion.claculate_region_accuracy(outputs, targets)
                            batch_accuracy = criterion.calculate_country_accuracy(outputs, targets)
                            
                            # Keep track of the number of samples in the current batch
                            batch_size = inputs.size(0)
                            
                            total_test_region_accuracy += batch_region_accuracy * batch_size
                            total_test_accuracy += batch_accuracy * batch_size
                            total_samples += batch_size

                    # Calculate average accuracies over all samples
                    avg_test_region_accuracy = total_test_region_accuracy / total_samples
                    avg_test_accuracy = total_test_accuracy / total_samples

                    print('Test Accuracy: {:.4f}, Test Regional Accuracy: {:.4f}'.format(avg_test_accuracy, avg_test_region_accuracy))
            finally:
                tracker.stop()