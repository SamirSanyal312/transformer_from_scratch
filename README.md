# Transformer from Scratch: German-to-English Translation

This project implements a Transformer-based neural machine translation model from scratch using Python and PyTorch. The model is trained on a German-English parallel dataset and learns to translate German sentences into English.

The main goal of this project is to understand how the Transformer architecture works internally, including tokenization, vocabulary creation, embeddings, positional encoding, multi-head attention, encoder-decoder layers, masking, training, evaluation, and translation generation.

---

## Project Description

Transformers are one of the most important architectures in modern Natural Language Processing and Large Language Models. Instead of relying on recurrent networks, Transformers use attention mechanisms to understand relationships between tokens in a sequence.

In this project, a Transformer model is built step by step from scratch for a machine translation task. The source language is German, and the target language is English.

The notebook demonstrates the full workflow of a sequence-to-sequence translation system:

- Load compressed German-English dataset files
- Tokenize German and English sentences
- Build source and target vocabularies
- Convert text into numerical token IDs
- Create padded batches for training
- Implement positional encoding
- Implement multi-head attention
- Build encoder and decoder layers
- Train the Transformer model
- Evaluate the model using loss and perplexity
- Generate sample English translations from German input sentences

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── transformer_code_from_scratch.ipynb
└── data/
    ├── train.de.gz
    ├── train.en.gz
    ├── val.de.gz
    ├── val.en.gz
    ├── test_2016_flickr.de.gz
    └── test_2016_flickr.en.gz
Dataset

The dataset contains parallel German-English sentence pairs.

The German files are used as the source language, and the English files are used as the target language.

Expected dataset files:

File	Description
train.de.gz	German training sentences
train.en.gz	English training sentences
val.de.gz	German validation sentences
val.en.gz	English validation sentences
test_2016_flickr.de.gz	German test sentences
test_2016_flickr.en.gz	English test sentences

Dataset statistics used in this project:

Split	German Sentences	English Sentences
Train	29,000	29,000
Validation	1,014	1,014
Test	1,000	1,000
Model Architecture

The Transformer model includes the following components:

Token embeddings
Sinusoidal positional encoding
Multi-head self-attention
Encoder layers
Decoder layers
Encoder-decoder attention
Position-wise feed-forward networks
Source padding mask
Target causal mask
Final linear projection layer

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

The validation loss decreased from 3.731 to 2.820, and the validation perplexity decreased from 41.714 to 16.773. This shows that the model successfully learned meaningful translation patterns over the training process.

Sample Translations
Example 1
Source:
ein mann mit einem orangefarbenen hut , der etwas anstarrt .

Target:
a man in an orange hat starring at something .

Predicted:
a man wearing an orange hat is talking to a orange hat .
Example 2
Source:
ein boston terrier läuft über saftig-grünes gras vor einem weißen zaun .

Target:
a boston terrier is running on lush green grass in front of a white fence .

Predicted:
a white and white dog runs across the grass in front of a white grass .
Example 3
Source:
ein mädchen in einem karateanzug bricht ein brett mit einem tritt .

Target:
a girl in karate uniform breaking a stick with a front kick .

Predicted:
a girl in a pink dress is eating a microphone .

The model generates grammatically structured English sentences, but some predictions are still semantically inaccurate. This is expected because the dataset is relatively small and the model was trained for only 5 epochs.

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
3. Upgrade pip
python -m pip install --upgrade pip
4. Install Required Packages
pip install torch numpy spacy tqdm
5. Download spaCy Language Models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
6. Add the Dataset Files

Create a folder named data and place all .gz dataset files inside it.

data/
├── train.de.gz
├── train.en.gz
├── val.de.gz
├── val.en.gz
├── test_2016_flickr.de.gz
└── test_2016_flickr.en.gz
7. Run the Notebook

Open the notebook:

transformer_code_from_scratch.ipynb

Then run all cells from top to bottom.

Local Path Configuration

When running the notebook locally, the dataset paths should be configured like this:

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

unless the notebook is being run inside Google Colab.

Key Concepts Covered

This project demonstrates the following Transformer and NLP concepts:

Tokenization
Vocabulary building
Word embeddings
Positional encoding
Self-attention
Multi-head attention
Encoder-decoder architecture
Source masking
Target causal masking
Teacher forcing
Sequence-to-sequence learning
Loss and perplexity evaluation
Greedy decoding for translation
Results and Observations

The model showed consistent improvement across all 5 epochs. Both training loss and validation loss decreased, showing that the model was learning from the dataset.

However, the generated translations are not perfect. In several examples, the model produces fluent English sentences but does not always preserve the exact meaning of the German input. This is a common limitation when training a small Transformer model on a relatively limited dataset.

The results show that the implementation is working correctly and that the model has learned useful translation patterns.

Limitations

The current implementation has the following limitations:

The dataset is relatively small for machine translation.
The model was trained for only 5 epochs.
Greedy decoding is used for inference.
BLEU score evaluation is not included.
Some predictions are fluent but semantically incorrect.
The model is built for educational understanding rather than production use.
Future Improvements

Possible improvements include:

Train the model for more epochs
Add BLEU score evaluation
Use beam search decoding instead of greedy decoding
Save and load model checkpoints
Tune hyperparameters
Increase dataset size
Add attention visualization
Compare results with pretrained translation models
Author

Samir Sanyal
Master's Student in Computer Science
Indiana University Bloomington

License

This project is licensed under the MIT License. See the LICENSE file for details.