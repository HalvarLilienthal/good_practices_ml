from concurrent.futures import ThreadPoolExecutor

import clip
import torch
import pandas as pd
import sys

sys.path.append('.')
import PIL
import numpy as np
from utils import load_dataset
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import torch.nn.functional as F


def save_dataframe_in_batches(df, batch_size, base_filename):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    for i in range(num_batches):
        batch = df[i * batch_size:(i + 1) * batch_size]
        batch.to_csv(f"{base_filename}_{i}.csv", index=False)


def improved_save_dataframe_in_batches(df, batch_size, base_filename):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

    save_batch = lambda df, filename: df.to_csv(filename, index=False)

    num_workers = 8
    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_batches):
            batch = df.iloc[i * batch_size:(i + 1) * batch_size]
            tasks.append(executor.submit(save_batch, batch, f"{base_filename}_{i}.csv"))

    for task in tasks:
        task.result()  # Warten auf alle parallelen Schreibvorgänge


def calculate_distances(image_embedding, prompt_embeddings: list):
    """ Calculate the cosine similarity between the image embedding and the prompt embeddings

    Args:
        embedding_str (tensor): The image embedding as a tensor
        prompt_embeddings (list): List of prompt embeddings

    Returns:
        np.array: The model input
    """
    image_embedding_values = np.array(image_embedding.flatten().tolist()).reshape(1, -1)

    prompt_distances = []
    # Reshape the vectors to be 2D arrays for sklearn's cosine_similarity
    # image_embedding = image_embedding.reshape(1, -1)
    for prompt_embedding in prompt_embeddings:
        prompt_embedding_values = np.array(prompt_embedding.flatten().tolist()).reshape(1, -1)
        # Calculate Cosine Similarity         
        prompt_distances.append(cosine_similarity(image_embedding_values, prompt_embedding_values)[0, 0])
    np.savetxt("prompt_distances_cpu.csv", prompt_distances, delimiter=",", fmt="%.4f")
    model_input = np.concatenate((image_embedding_values[0], np.array(prompt_distances))).astype(np.float32)
    np.savetxt("model_input_cpu.csv", model_input, delimiter=",", fmt="%.4f")
    return model_input.tobytes()


def gpu_calculate_distances(image_embedding: torch.Tensor, prompt_embeddings: torch.Tensor) -> bytes:
    """ A GPU accelerated version of the calculation of the cosine similarity between the image embedding and the prompt
    embeddings

    Args:
        image_embedding (torch.Tensor): The image embedding as a tensor
        prompt_embeddings (list): List of prompt embeddings as a tensor

    Returns:
        bytes: Das Model-Input als Binärdaten
    """
    with torch.inference_mode():
        image_embedding = image_embedding.view(1, -1)
        prompt_distances = F.cosine_similarity(image_embedding, prompt_embeddings, dim=1)
        model_input = torch.cat((image_embedding.view(-1), prompt_distances)).float()

    return model_input.cpu().numpy().tobytes()


def generate_embeddings(REPO_PATH, DATA_PATH, gpu: bool = False):
    """
    Generates embeddings for the geoguessr, tourist and aerial datasets and saves them to csv files

    Args:
        REPO_PATH (str): The path to the repository
        DATA_PATH (str): The path to the data folder
    """
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    model, preprocessor = clip.load("ViT-B/32", device=device)

    # load image data
    geoguessr_df = load_dataset.load_data(f'{DATA_PATH}/geoguessr')
    tourist_df = load_dataset.load_data(f'{DATA_PATH}/tourist')
    aerial_df = load_dataset.load_data(f'{DATA_PATH}/aerial')

    # generate Prompts
    country_list = pd.read_csv(f'{REPO_PATH}/utils/country_list/country_list_region_and_continent.csv')[
        "Country"].to_list()
    country_prompt = list(map((lambda x: f"This image shows the country {x}"), country_list))

    with torch.inference_mode():
        # generate image embeddings
        geoguessr_df["Embedding"] = geoguessr_df["path"].apply(
            lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
        tourist_df["Embedding"] = tourist_df["path"].apply(
            lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))
        aerial_df["Embedding"] = aerial_df["path"].apply(
            lambda path: model.encode_image(preprocessor(PIL.Image.open(path)).unsqueeze(0).to(device)))

        # generate prompt embeddings
        simple_tokens = clip.tokenize(country_list)
        promt_token = clip.tokenize(country_prompt)

        simple_tokens = simple_tokens.to(device)
        promt_token = promt_token.to(device)
        model = model.to(device)

        simple_embedding = model.encode_text(simple_tokens)
        prompt_embedding = model.encode_text(promt_token)

    # Use either CPU (numpy based) or GPU (pytorch based) version of the cosine similarity calculation
    distance_calculation = calculate_distances if not gpu else gpu_calculate_distances

    # generate model inputs, by appending distances to the prompt embeddings
    geoguessr_df["model_input"] = geoguessr_df["Embedding"].apply(
        lambda x: distance_calculation(x, prompt_embedding))
    aerial_df["model_input"] = aerial_df["Embedding"].apply(lambda x: distance_calculation(x, prompt_embedding))
    tourist_df["model_input"] = tourist_df["Embedding"].apply(
        lambda x: distance_calculation(x, prompt_embedding))

    # save image embeddings
    improved_save_dataframe_in_batches(geoguessr_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/geoguessr_embeddings")
    improved_save_dataframe_in_batches(tourist_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/tourist_embeddings")
    improved_save_dataframe_in_batches(aerial_df, 2000, f"{REPO_PATH}/CLIP_Embeddings/Image/aerial_embeddings")

    # save prompt embeddings
    torch.save(simple_embedding, f'{REPO_PATH}/CLIP_Embeddings/Prompt/prompt_simple_embedding.pt')
    torch.save(prompt_embedding, f'{REPO_PATH}/CLIP_Embeddings/Prompt/prompt_image_shows_embedding.pt')


if __name__ == "__main__":
    """Generates embeddings for the geoguessr, tourist and aerial datasets and saves them to csv files
    """
    parser = argparse.ArgumentParser(description='Generate Embeddings')
    parser.add_argument('--yaml_path', metavar='str', required=False,
                        help='The path to the yaml file with the stored paths', default='../paths.yaml')
    parser.add_argument('-d', '--debug', action='store_true',
                        required=False, help='Enable debug mode', default=False)
    parser.add_argument('--gpu', help="Use an GPU accelerated version of this script.", action='store_true',
                        default=False)
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path']
        DATA_PATH = paths['data_path']
        generate_embeddings(REPO_PATH, DATA_PATH, args.gpu)
