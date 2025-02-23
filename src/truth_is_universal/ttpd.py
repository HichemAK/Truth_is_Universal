import glob
import json
import os
import os.path as osp
from typing import Iterable
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd



STORAGE_FOLDER = os.envrion.get('STORAGE_FOLDER', "./")

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs

def load_dataset(dataset_name) -> tuple[list[str], list[int], list[float]]:
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(osp.join(osp.dirname(__file__),f"datasets/{dataset_name}.csv"))
    statements = dataset['statement'].tolist()
    labels = dataset['label'].tolist()
    polarity = -1.0 if 'neg_' in dataset_name else 1.0
    polarities = [polarity]*len(labels)
    return statements, labels, polarities

def extract_layers(model) -> torch.nn.Module:
    pass

def collect_activations(model, tokenizer, statements : list[str], device) -> torch.Tensor:
    inp = tokenizer(statements, return_tensors='pt')
    acts = model(inp.input_ids, activation_mask=inp.activation_mask, output_hidden_states=True).hidden_states[:,-1].cpu()
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

    statements, labels, polarities = zip(*[(statement, label, polarity) for dataset in datasets for statement, label, polarity in load_dataset(dataset)])
    return statements, labels, polarities, datasets

def batched(iterable : Iterable, batch_size: int) -> Iterable[list]:
    l = []
    for x in iterable:
        if len(l) >= batch_size:
            yield l
            l = []
        l.append(x)
    yield l


class TTPDRegressor:
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe
    
    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return torch.tensor(self.LR.predict(acts_2d))
    
    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d

class TTPD:
    FOLDER_NAME = "TTPDTruthDirections"
    def __init__(self, model_name : str, precision=16):
        self.model_name = model_name
        self.truth_vector: list[float] = None
        self.polarity_vector: list[float] = None

        assert precision in [16,32]
        self.precision = precision
        self.torch_dtype = torch.bfloat16 if precision == 16 else torch.float32

    @property
    def path(self) -> str:
        return osp.join(STORAGE_FOLDER, TTPD.FOLDER_NAME, "vectors__%s.json" % self.model_name)

    def is_built(self) -> bool:
        return osp.exists(self.path)
    
    def load(self) -> None:
        if not self.is_built():
            raise Exception('Call self.build() first!')
        
        vectors = json.load(open(self.path))
        self.truth_vector = vectors['truth']
        self.polarity_vector = vectors['polarity']
    
    @torch.no_grad()
    def build(self, batch_size: int, device_map: str, force=False) -> None:
        if self.is_built() and not force:
            raise Exception("TTDP Truth Directions were already computed for the model %s \n Use force=True to rebuild and replace previous vectors" % self.model_name)
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, torch_dtype=self.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = 'left'

        device = next(model.named_parameters().values()).device

        statements, labels, datasets_names = load_statements(["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors", "animal_class",
                  "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]) # The same dataset used in the paper for lie detection.
        batched_statements_iter = batched(statements, batch_size)
        activations = []
        for batch in tqdm(batched_statements_iter, "Generating activations", total=len(statements) // batch_size):
            acts = collect_activations(model, tokenizer, batch, device)
            activations.append(acts)

        # print("Saving activations...")
        activations = torch.cat(activations, dim=0)
        # for layer in range(activations.shape[1]):
        #     torch.save(activations[:, layer], osp.join(self.path, f"layer_{layer}.pt"))
        # print("Activations saved!")


        # Find best layer
        within_class_variances = []
        between_class_variances = []
        labels = np.array(labels)
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
        best_layer = np.argmax(layer2perf)

        # Compute truth directions
        activations_best_layer = activations[:, best_layer]
        activations_best_layer_centered = activations_best_layer - activations_best_layer.mean(1)
        acts_centered, _, labels, polarities = collect_training_data(cv_train_sets, train_set_sizes, model_family,
                                                          model_size, model_type, layer)
        
        # Fit model
        t_g, t_p = learn_truth_directions(acts_centered, labels, polarities)