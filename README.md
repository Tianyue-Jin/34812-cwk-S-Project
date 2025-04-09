
# COMP34812 Coursework Submission - Group 9

This repository contains our submission for the COMP34812 Shared Task coursework. We have implemented and evaluated two models corresponding to the **Evidence Detection** (ED) track:

- **Category A**: Fine-tuned SVM model using TF-IDF features.
- **Category C**: Fine-tuned Transformer-based model using the base model `microsoft/deberta-v3-base`.

---

## Deliverables

### Deliverable 1: Predictions

Our predictions on the test dataset are saved in:

- `Group_9_A.csv` — Predictions from the fine-tuned **SVM** model (Category A).
- `Group_9_C.csv` — Predictions from the fine-tuned **DeBERTa-v3** model (Category C).

Both files follow the required format and naming convention.

---

### Deliverable 2: Code, Models, and Demo Notebooks

Our submission includes:

#### Training Notebooks:
- `Category_A_SVM_fine_tuning.ipynb` — Jupyter notebook for tuning the SVM model.
- `Category_C_DeBERTa_v3.ipynb` — Jupyter notebook for training and evaluating the DeBERTa model.

#### Demo Notebook:
- `Demo_Code.ipynb` — Code to run inference and generate predictions using the saved SVM model and the DeBERTa model.

#### Saved Models:
- `svm_best_model.joblib` — Saved GridSearchCV SVM pipeline.
- `deberta_v3_model/`  — Trained DeBERTa-v3 checkpoint directory (includes tokenizer, config, and safetensors).

**Note:** All models are stored on the cloud via Google Drive due to size constraints and can be accessed via: 
https://drive.google.com/drive/folders/1X5vCOMxn1Pi56gSiRixOX-M19jDVfMhX?usp=sharing

---

### Deliverable 3: Model Cards

Each model has an associated model card in markdown format, describing:

- Training objectives
- Architecture and hyperparameters
- Datasets used (dev.csv and test.csv)
- Evaluation metrics
- Risks and limitations
- Hardware/software setup

The following markdown model cards were generated based on coursework templates and published on Hugging Face:

- [`svm_model_card.md`](https://huggingface.co/tyjin020726/SVM_Evidence_Detection)
- [`deberta_model_card.md`](https://huggingface.co/tyjin020726/DeBERTaV3_Evidence_Detection)

Each card contains structured metadata, performance metrics, and references to model architecture and base repositories.

---

### Deliverable 4: Poster (PDF)

The poster is provided in landscape format and outlines:

- Our overall solution approach
- Comparison between SVM and DeBERTa
- Performance metrics and charts
- Visual explanation of our methodology

 `COMP34812_Poster_Group9.pdf`

---

## Evaluation Metrics

We evaluated both models on the development set using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score (Macro, Weighted)
- Matthews Correlation Coefficient (MCC)

Model cards were created for both models using HuggingFace’s model card template and include detailed summaries, metrics, and results.

---

## Attribution of Sources

We used the following public resources:

- [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) — Pretrained base model from Hugging Face.
- Hugging Face Transformers library for DeBERTa fine-tuning and loading.
- Scikit-learn's `GridSearchCV`, `Pipeline`, and SVM (`SVC`) for classical ML model tuning.

We acknowledge reuse of standard techniques and utilities but designed the model selection, training pipelines, and demos ourselves.

---

## Use of Generative AI Tools

We used **ChatGPT** by OpenAI to assist with the following tasks:

- Refining Python and markdown documentation.
- Clarifying library usage and debugging errors.
- Summarizing evaluation metrics and preparing sections of this README, model cards, and poster.

All generated content was reviewed, modified, and validated by the team to ensure accuracy and full understanding.

---

## Developers

- **Sicheng Pan**
- **Tianyue Jin**

_University of Manchester, COMP34812 Coursework 2024–25_

---
