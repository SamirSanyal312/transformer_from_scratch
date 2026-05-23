# Transformer from Scratch: German to English Translation

This project implements a Transformer-based neural machine translation model from scratch using PyTorch. The model is trained to translate German sentences into English using a parallel German-English dataset.

The goal of this project is to understand the internal working of the Transformer architecture, including embeddings, positional encoding, multi-head attention, encoder-decoder layers, masking, training, and inference.

---

## Project Overview

This notebook builds a complete Transformer model without relying on high-level translation libraries. It covers the full machine learning pipeline:

- Loading compressed German-English dataset files
- Tokenizing source and target sentences
- Building source and target vocabularies
- Numericalizing text into token IDs
- Creating padded mini-batches
- Implementing positional encoding
- Implementing multi-head self-attention
- Building encoder and decoder layers
- Training the Transformer model
- Evaluating using loss and perplexity
- Generating sample German-to-English translations

---

## Dataset

The project uses German-English parallel text files stored in compressed `.gz` format.

Expected dataset structure:

```text
data/
├── train.de.gz
├── train.en.gz
├── val.de.gz
├── val.en.gz
├── test_2016_flickr.de.gz
└── test_2016_flickr.en.gz

The German files are used as the source language, and the English files are used as the target language.

Dataset statistics used in this project:

Split	German Sentences	English Sentences
Train	29,000	29,000
Validation	1,014	1,014
Test	1,000	1,000
Model Architecture

The implemented Transformer includes:

Source and target word embeddings
Sinusoidal positional encoding
Multi-head attention
Position-wise feed-forward network
Encoder layers
Decoder layers
Source padding mask
Target causal mask
Final linear projection to target vocabulary

Model summary from the training run:

Source vocab size: 18669
Target vocab size: 9797
Trainable parameters: 13,758,789
Training Results

The model was trained for 5 epochs.

Epoch	Train Loss	Train PPL	Validation Loss	Validation PPL
1	4.539	93.628	3.731	41.714
2	3.577	35.769	3.345	28.355
3	3.261	26.081	3.115	22.537
4	3.043	20.977	2.976	19.600
5	2.874	17.702	2.820	16.773

The validation loss and perplexity consistently decreased across epochs, showing that the model successfully learned meaningful translation patterns.

Sample Translations

Example 1:

Source:
ein mann mit einem orangefarbenen hut , der etwas anstarrt .

Target:
a man in an orange hat starring at something .

Predicted:
a man wearing an orange hat is talking to a orange hat .

Example 2:

Source:
ein boston terrier läuft über saftig-grünes gras vor einem weißen zaun .

Target:
a boston terrier is running on lush green grass in front of a white fence .

Predicted:
a white and white dog runs across the grass in front of a white grass .

Example 3:

Source:
ein mädchen in einem karateanzug bricht ein brett mit einem tritt .

Target:
a girl in karate uniform breaking a stick with a front kick .

Predicted:
a girl in a pink dress is eating a microphone .

The model generates grammatically structured English sentences, but some translations remain semantically inaccurate. This is expected because the model was trained on a relatively small dataset and for only a few epochs.

Technologies Used
Python
PyTorch
NumPy
spaCy
Jupyter Notebook
tqdm
Setup Instructions
1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>
2. Create a Virtual Environment

For Windows PowerShell:

python -m venv env2
.\env2\Scripts\activate

For macOS/Linux:

python -m venv env2
source env2/bin/activate
3. Install Dependencies
pip install torch numpy spacy tqdm
4. Download spaCy Language Models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
5. Place Dataset Files

Create a data folder and place the .gz files inside it:

data/
├── train.de.gz
├── train.en.gz
├── val.de.gz
├── val.en.gz
├── test_2016_flickr.de.gz
└── test_2016_flickr.en.gz
6. Run the Notebook

Open the notebook:

transformer_code_from_scratch.ipynb

Then run all cells from top to bottom.

Important Notes

If you are running this locally, make sure the dataset paths are set like this:

from pathlib import Path

DATA_DIR = Path("data")

train_de_path = DATA_DIR / "train.de.gz"
train_en_path = DATA_DIR / "train.en.gz"

val_de_path = DATA_DIR / "val.de.gz"
val_en_path = DATA_DIR / "val.en.gz"

test_de_path = DATA_DIR / "test_2016_flickr.de.gz"
test_en_path = DATA_DIR / "test_2016_flickr.en.gz"

Avoid using Google Colab-specific paths such as:

/content/data/train.de.gz

unless you are running the notebook inside Google Colab.

Key Learnings

This project helped demonstrate how the Transformer architecture works internally. Instead of using prebuilt translation APIs or high-level frameworks, the model was implemented step by step to understand:

How attention scores are computed
How multi-head attention captures different relationships between tokens
How positional encoding gives sequence order information
How encoder-decoder attention supports translation
How masking prevents the decoder from seeing future tokens
How loss and perplexity measure translation model performance
Limitations

The current model has some limitations:

The dataset is relatively small for machine translation.
The model was trained for only 5 epochs.
Greedy decoding is used during translation.
No BLEU score evaluation is currently included.
Some generated translations are fluent but semantically incorrect.

Future improvements could include:

Training for more epochs
Adding BLEU score evaluation
Using beam search decoding
Increasing dataset size
Tuning model hyperparameters
Saving and loading trained checkpoints
Author

Samir Sanyal

Master's Student in Computer Science
Indiana University Bloomington