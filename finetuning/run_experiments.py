import argparse
import time

from codecarbon import OfflineEmissionsTracker

import create_datasets_from_embddings
import sys
sys.path.append("./../")
import finetuning.model.model_trainer as trainer
import yaml

if __name__ == "__main__":
    """Runs the initial CLIP experiments
    """
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths', default='../paths.yaml')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path']
        REPO_PATH = paths['repo_path']
        #create_datasets_from_embddings.create_datasets_from_embddings(REPO_PATH, seed=1234)
        seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]
        all_start = time.time()
        for seed in seeds:
            start = time.time()

            tracker = OfflineEmissionsTracker(
                experiment_id=f"{seed}",
                country_iso_code="DEU",
                measure_power_secs=5,
                project_name="run_experiments.py",
                tracking_mode="process",
                # allow_multiple_runs=True,    # Set this to True to allow multiple instances of codecarbon to run at the same time
            )

            tracker.start()

            try:
                trainer.create_and_train_model(REPO_PATH, seed)
                print("Train Time for seed", seed, "is", time.time() - start)
            finally:
                tracker.stop()
        print("Full Train Time is", time.time() - all_start)