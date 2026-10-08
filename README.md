# Identifying LLMs from Their Responses

**CMPSC 448 Midterm Project**  
**Author and team leader:** Sangaa Chatterjee (individual project)

## Overview

For this project, I investigated whether a classifier can identify which language-model family produced a response. I used the TXD-22 dataset and its ChatGPT, Claude, and Gemini source labels. I implemented and evaluated a convolutional neural network (CNN) and a bidirectional long short-term memory network (BiLSTM) in PyTorch.

I investigated the two required questions: **RQ1**, whether a model can identify the source from the response alone, and **RQ2**, whether including the user's prompt helps classification. I also explored **RQ4** for extra credit by measuring writing characteristics and retraining models with punctuation removed or model/vendor names masked. I did not attempt the optional cross-domain RQ3.

## Project files

| File or folder | Purpose |
| --- | --- |
| `report.pdf` | Write-up on the project |
| `experiment.py` | Data cleaning and splitting, CNN/BiLSTM definitions, training, testing, and experiment outputs. |
| `download_data.py` | Downloads the public TXD-22 archive into `data/`. |
| `predict.py` | Runs inference on a text file using a locally available output-only checkpoint. |
| `verify_results.py` | Checks saved experimental metrics, confusion matrices, prompt grouping, class balance, and padding behavior. |
| `requirements.txt` | Python dependencies. |
| `team.json` | Individual project/team-leader information. |
| `data/README.md` | Dataset provenance and reuse notes. |
| `results_reproduction/` | Saved results: summary and audit data, prediction CSVs, model-specific metrics JSONs, split manifest, and experiment configuration. |
| `.gitignore` | Excludes raw downloaded data, trained `.pt` checkpoints, virtual environments, and Python cache files from Git. |

The original dataset is not redistributed in this repository. Training creates `.pt` model checkpoints in the results folder, but `.gitignore` excludes those checkpoints from GitHub. Checkpoints may still be present in a local working folder or a separately created ZIP. The saved metrics and prediction files can be inspected without the checkpoints.

## Requirements and setup

I used Python 3.12.7 for the reported experiments. The exact versions recorded in `results_reproduction/run_config.json` should be consulted for an exact reproduction; installing the dependencies in `requirements.txt` may install newer compatible releases.

Open a terminal **inside the `llm_fingerprints` project folder**.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked, use ` .\.venv\Scripts\python.exe ` in place of `python` for the commands below.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the experiments

First, download the public dataset:

```bash
python download_data.py
```

Then run the complete training and evaluation pipeline:

```bash
python experiment.py --out results_reproduction --epochs 5 --max-len 128 --seeds 42
```

This trains both neural architectures in six input conditions each, then evaluates a TF-IDF/logistic-regression baseline. The pipeline saves JSON metrics, held-out predictions, split information, training histories, and local model checkpoints. Training may take significant time on a CPU. It does not require an LLM API key.

The source archive version can change. Compare its hash with `results_reproduction/data_audit.json` before claiming an exact reproduction of the original run.

## Verify the saved results

The provided `results_reproduction/` folder already contains the saved results needed for verification. From the project root, after installing the requirements, run:

```bash
python verify_results.py
```

The script checks that:

1. All 12 CNN/BiLSTM experimental result files are present.
2. Prediction CSVs reproduce the reported accuracy, macro-F1, and confusion matrices.
3. Each normalized prompt belongs to only one of the train/validation/test splits.
4. Retained prompt groups have one response from each of the three model families.
5. CNN and BiLSTM outputs do not change when extra padding is appended to a test sequence.

Successful execution prints a JSON summary with `experiments_verified` equal to `12` and the remaining checks set to `true`. The script also writes `results_reproduction/verification.json`. **It verifies the existing saved results; it does not retrain the models or independently authenticate the original dataset labels.**

If you rename the results directory, you must update the `r = Path('results_reproduction')` line in `verify_results.py`. No edit is needed when using the folder name in this repository.

## Models and experimental design

After data cleaning and filtering, the experiment retained **8,901 responses from 2,967 shared prompts**, balanced across the three source labels. Normalized prompts were split into training (2,076 prompt groups), validation (445), and test (446), so the same normalized prompt could not appear in multiple splits. Vocabulary construction used training data only, and the best epoch was selected using validation macro-F1.

- **CNN:** Learned 64-dimensional embeddings; convolution widths 3, 4, and 5 with 64 filters each; masked max pooling; dropout 0.3; three-class output layer.
- **BiLSTM:** Learned 64-dimensional embeddings; one bidirectional LSTM with hidden size 64 in each direction; packed sequences; dropout 0.3; three-class output layer.

Both architectures used Adam (learning rate 0.001, weight decay 0.0001), batch size 64, gradient clipping at 1.0, and seed 42. Models were trained for up to five epochs per condition.

The six neural input conditions were prompt-only, response-only (128 tokens), combined prompt-and-response, response-only (64-token matched control), punctuation-stripped plain text, and model/vendor-name-masked text.

## Main test results

| Input condition | CNN accuracy | BiLSTM accuracy |
| --- | ---: | ---: |
| Prompt only | 33.3% | 33.3% |
| Response only, 128 tokens | 83.3% | 70.3% |
| Prompt + response | 75.7% | 68.4% |
| Response only, 64 tokens | 76.8% | 73.2% |
| Plain text | 74.4% | 66.9% |
| Model/vendor names masked | 82.6% | 73.8% |

The separate TF-IDF/logistic-regression baseline achieved **92.7%** accuracy. It uses a different text budget, so it is not a strictly matched comparison to the neural architectures.

In RQ2, I included the 64-token response-only control because the combined input reserves tokens for the prompt and otherwise provides less response text than the 128-token baseline. The experiments did not show an improvement from adding the prompt under this matched comparison.

## Optional inference

If an output-only checkpoint exists locally, put the text to classify in a UTF-8 text file, for example `sample.txt`, and run:

```bash
python predict.py --checkpoint results_reproduction/CNN_output_s42.pt --text-file sample.txt
```

The command prints three closed-set class scores. The checkpoint is not available after a fresh Git clone unless you retrain the models or provide a local checkpoint; `.pt` files are Git-ignored. These scores are not proof of authorship and are not guaranteed to be calibrated probabilities.

## Limitations and responsible use

The dataset's family labels, exact model versions, prompts used during original collection, and generation settings were not independently authenticated. The experiment used one curated dataset, one training seed, and relatively short neural input sequences. The test split avoids exact normalized-prompt overlap, but related templates and source-specific artifacts could still make classification easier. Results may not transfer to different datasets, new model versions, human-written text, or rewritten outputs.

The source archive was obtained from TXD-22 on Kaggle (listed version 2, retrieved October 8, 2026). Its reuse license was listed as unknown, so I have not redistributed the raw source text and would verify permissions before sharing the dataset or trained checkpoints.

## References

- [TXD-22 dataset, Kaggle](https://www.kaggle.com/datasets/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset)
- [PyTorch `Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [PyTorch `LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch tutorial: NLP from scratch](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

For full methods, figures, confidence intervals, per-class metrics, and reflections, see **`report.pdf`**.
