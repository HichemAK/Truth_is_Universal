import glob
import json
import os
import os.path as osp
import shutil
from typing import Iterable
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from .regressor import TTPD as TTPDRegressor


STORAGE_FOLDER = os.environ.get('STORAGE_FOLDER', "./")

def load_dataset(dataset_name) -> tuple[list[str], list[int], list[float]]:
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(osp.join(osp.dirname(__file__),f"datasets/{dataset_name}.csv"))
    statements = dataset['statement'].tolist()
    labels = dataset['label'].tolist()
    polarity = -1.0 if 'neg_' in dataset_name else 1.0
    polarities = [polarity]*len(labels)
    return statements, labels, polarities, len(labels)


def collect_activations(model, tokenizer, statements : list[str], device) -> torch.Tensor:
    inp = tokenizer(statements, return_tensors='pt', padding=True).to(device)
    hs = torch.stack(model(inp.input_ids, attention_mask=inp.attention_mask, output_hidden_states=True).hidden_states, dim=1)
    acts = hs[:,1:,-1].cpu()
    return acts

def load_statements(datasets: str | list[str]) -> tuple[Iterable[str], list[int], list[str]]:
    if isinstance(datasets, str):
        datasets = [datasets]
    if datasets == ['all_topic_specific']:
        datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
                    'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
                    'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
                    'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
                    'larger_than', 'smaller_than', "cities_de", "neg_cities_de", "sp_en_trans_de", "neg_sp_en_trans_de", "inventors_de", "neg_inventors_de", "animal_class_de",
                  "neg_animal_class_de", "element_symb_de", "neg_element_symb_de", "facts_de", "neg_facts_de"]
    if datasets == ['all']:
        datasets = []
        for file_path in glob.glob(osp.join(osp.dirname(__file__), 'datasets/**/*.csv'), recursive=True):
            dataset_name = os.path.relpath(file_path, 'datasets').replace('.csv', '')
            datasets.append(dataset_name)

    datasets_sizes = []

    all_statements, all_labels, all_polarities  = [],[],[]
    
    for dataset in datasets:
        statements, labels, polarities, size = load_dataset(dataset)
        datasets_sizes.append(size)
        all_statements.extend(statements)
        all_labels.extend(labels)
        all_polarities.extend(polarities)
    return all_statements, all_labels, all_polarities, datasets, datasets_sizes

def batched(iterable : Iterable, batch_size: int) -> Iterable[list]:
    l = []
    for x in iterable:
        if len(l) >= batch_size:
            yield l
            l = []
        l.append(x)
    yield l


def compute_layer2perf(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    within_class_variances = []
    between_class_variances = []
    for layer_nr in range(activations.shape[1]):
        acts = activations[:, layer_nr] 
        # Calculate means for each class
        false_stmnt_ids = labels == 0
        true_stmnt_ids = labels == 1

        false_acts = acts[false_stmnt_ids]
        true_acts = acts[true_stmnt_ids]

        mean_false = false_acts.mean(dim=0)
        mean_true = true_acts.mean(dim=0)

        # Calculate within-class variance
        within_class_variance_false = false_acts.var(dim=0).mean()
        within_class_variance_true = true_acts.var(dim=0).mean()
        within_class_variances.append((within_class_variance_false + within_class_variance_true).item() / 2)

        # Calculate between-class variance
        overall_mean = acts.mean(dim=0)
        between_class_variances.append(((mean_false - overall_mean).pow(2) 
                                        + (mean_true - overall_mean).pow(2)).mean().item() / 2)

    layer2perf = np.array(between_class_variances) / np.array(within_class_variances)
    return layer2perf

def center_activations(activations: torch.Tensor, dataset_sizes: list[int]) -> torch.Tensor:
    activations = activations.clone()
    cumsum_sizes = np.array(dataset_sizes).cumsum()
    for i in range(len(dataset_sizes)):
        start = cumsum_sizes[i]
        end = start + dataset_sizes[i]
        activations[start:end] = activations[start:end] - activations[start:end].mean(0).unsqueeze(0)
    return activations

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class TTPD:
    FOLDER_NAME = "TTPDTruthDirections"
    def __init__(self, model_name : str, precision=16, model=None, tokenizer=None, device_map="auto"):
        self.model_name = model_name
        self.t_g: torch.Tensor = None
        self.t_p: torch.Tensor = None
        self.model = model
        self.tokenizer = tokenizer
        self.device_map = device_map

        assert precision in [16,32]
        self.precision = precision
        self.torch_dtype = torch.bfloat16 if precision == 16 else torch.float32

    @property
    def path(self) -> str:
        return osp.join(STORAGE_FOLDER, TTPD.FOLDER_NAME, "vectors__%s.json" % self.model_name.replace("/", "_"))

    def is_built(self) -> bool:
        return osp.exists(self.path)
    
    def load(self) -> None:
        if not self.is_built():
            raise Exception('Call self.build() first!')
        
        vectors = json.load(open(self.path))
        self.t_g = torch.tensor(vectors['truth'])
        self.t_p = torch.tensor(vectors['polarity'])
        self.best_layer = vectors["best_layer"]
    
    @torch.no_grad()
    def build(self, batch_size: int, force=False) -> None:
        if self.is_built() and not force:
            raise Exception("TTDP Truth Directions were already computed for the model %s \n Use force=True to rebuild and replace previous vectors" % self.model_name)
        if force:
            self.destroy()
        # Step 1: Collect activations
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token

        device = next(self.model.named_parameters())[1].device

        dataset_list = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors", "animal_class",
                  "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]
        statements, labels, polarities, _, dataset_sizes = load_statements(dataset_list) # The same dataset used in the paper for lie detection.
        batched_statements_iter = batched(statements, batch_size)
        activations = []
        for batch in tqdm(batched_statements_iter, "Generating activations", total=len(statements) // batch_size):
            acts = collect_activations(self.model, self.tokenizer, batch, device)
            activations.append(acts)

        activations = torch.cat(activations, dim=0)

        # Step 2: Find best layer
        labels = np.array(labels)
        best_layer_datasets = ['cities', 'neg_cities', 'sp_en_trans', 'neg_sp_en_trans']
        cumsum_sizes = np.array(dataset_sizes).cumsum()
        mask = np.zeros(activations.shape[0], dtype=np.bool)
        for ds in best_layer_datasets:
            idx = dataset_list.index(ds)
            start = cumsum_sizes[idx]
            end = start + dataset_sizes[idx]
            mask[start:end] = 1
        best_layer_activations = activations[mask]
        best_layer_labels = labels[mask]
        layer2perf = compute_layer2perf(best_layer_activations, best_layer_labels)
        best_layer = np.argmax(layer2perf)
        print("Layer2perf")
        print(layer2perf)
        print("Best layer index : %s" % best_layer)

        # Step 3: Compute truth directions
        activations_best_layer = activations[:, best_layer].float()
        # activations_best_layer_centered = center_activations(activations_best_layer, dataset_sizes)
        activations_best_layer_centered = activations_best_layer - activations_best_layer.mean(0)
        labels = torch.tensor(labels)
        polarities = torch.tensor(polarities)
        regressor = TTPDRegressor.from_data(activations_best_layer_centered, activations_best_layer, labels, polarities)

        # Step 4: Save truth directions
        t_g, t_p = regressor.t_g.tolist(), regressor.polarity_direc[0].tolist()
        os.makedirs(osp.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "truth": t_g,
                "polarity": t_p,
                "best_layer": best_layer 
            }, f, cls=NpEncoder)
        print("Truth directions saved in %s" % self.path)
        print("Build finished!")

    def project(self, statements: list[str]):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = 'left'
        device = next(self.model.named_parameters().values()).device

        outputs = collect_activations(self.model, self.tokenizer, statements, device)
        activations = outputs[:, self.best_layer]
        proj_t_g = activations @ self.t_g
        proj_p = activations @ self.t_p.T
        acts_2d = torch.cat((proj_t_g[:, None], proj_p), dim=1)
        return acts_2d

    def destroy(self) -> None:
        shutil.rmtree(osp.dirname(self.path), ignore_errors=True)