# Truth is Universal

This repository contains the code and datasets necessary to reproduce the results presented in the paper "Truth is Universal: Robust Detection of Lies in LLMs".

## Set-up
We recommend using conda for Python installation. While we used Python 3.11.9, other versions should be compatible.
It's advisable to create a new Python environment before installing the required packages.
Create and activate the environment:
```
conda create --name truth_is_universal python=3.11
conda activate truth_is_universal
```
Here, `python=3.11` is optional and other versions should be compatible as well. 
Navigate to your preferred repository location, then clone the repository, enter it, and install the requirements:
```
git clone git@github.com:sciai-lab/Truth_is_Universal.git
cd Truth_is_Universal
pip install -r requirements.txt
```
This repository provides all datasets used in the paper, but not the associated activation vectors due to their large size. You'll need to generate these activations before running any other code. This requires model weights from the Llama3, Llama2, or Gemma model family. We suggest obtaining these weights from <a href="https://huggingface.co/">Hugging Face</a> (e.g. Llama3-8B-Instruct <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">here</a>). Insert the paths to the weight folders into `config.ini`.
Then, run `generate_acts.py` to generate the activations. For example, to generate activations for the cities and neg_cities datasets for Llama3-8B-Instruct in layers 11 and 12:
```
python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers 11 12 --datasets cities neg_cities --device cuda:0
```
The model runs in float16 precision. Hence, at least 16GB of GPU RAM are required to run Llama3-8B.
The activations will be stored in the `acts` folder. You can generate the activations for all layers by setting `--layers -1`. You can generate the activations for all topics-specific datasets (defined in the paper) by setting `--datasets all_topic_specific` and for all datasets by setting `--datasets all`.

## Repository Structure
* `generate_acts.py`: For generating activations as described above.
* `utils.py`: Contains various helper functions, e.g. for loading the activations or for learning the truth directions.

Jupyter Notebooks:

* `truth_directions.ipynb`: Code for generating figures from the first four paper sections; from learning truth directions to exploring the dimensionality of the truth subspace. 
You need to generate the following activations to run this notebook (e.g. for Llama3-8B-Instruct):
```
python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers 12 --datasets all_topic_specific --device cuda:0
```
and
```
python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers -1 --datasets cities neg_cities sp_en_trans neg_sp_en_trans --device cuda:0
```

More to be added in the next couple of days!


## Credit
The DataManager class in `utils.py` and the script `generate_acts.py` are (up to some modifications) from the <a href="https://github.com/saprmarks/geometry-of-truth">Geometry of Truth GitHub repository by Samuel Marks.</a>
The datasets in the datasets folder were primarily collected from previous papers, all of which are referenced in our paper.

