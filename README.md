# Paper
[![arXiv](https://img.shields.io/badge/arXiv-2405.09335-b31b1b.svg?style=flat)](https://arxiv.org/abs/2405.09335)
[![Generic badge](https://img.shields.io/badge/LREC_COLING_2024-Link-GREEN.svg?style=flat)](https://aclanthology.org/2024.lrec-main.1153/)

# Installation
Clone the repo and install the required Python packages (preferably in a virtual environment) from `requirements_install_first.txt` first followed by `requirements.txt`.

# Download Data
You can load any dataset that is compatible with the datasets library.

To download the datasets of the few-shot MRQA benchmark, see [https://github.com/oriram/splinter#downloading-few-shot-mrqa-splits](https://github.com/oriram/splinter#downloading-few-shot-mrqa-splits).

<!-- # Run
There are scripts for training question generation (qg) models (run_training_qg_few_shot.sh), generating data and training RC models using synthetic data from qg models (run_qg_few_shot.sh & run_qg_zero_shot.sh) and training Prompting models for RC (run_rc_few_shot.sh). -->

# Data Generation

Data generation using Prompting is run using the run script with the qg argument, e.g.,
```bash
python run.py qg \
    --output_dir models/qg-squad-16 \
    --transformer t5-v1-1-large \
    --prompt-method soft \
    --template-idx 22 \
    --lang en \
    --num_worker 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --train-dataset sp-squad-16-0 \
    --do_eval \
    --eval-dataset sp-squad-dev::1 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 5 --logging_steps 5 \
    --save_steps 5 \
    --logging_first_step --early-stopping 130 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --optim adafactor \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --max_steps 200 \
    --train_chunking_mode token \
    --train_context_stride 100 \
    --train_context_size 450 \
    --eval_chunking_mode token \
    --eval_context_stride 100 \
    --eval_context_size 450 \
    --ft-model \
    --bf16_full_eval
```
for training a T5 v1.1 large model on a 16-split from the SQuAD subset from the few-shot MRQA benchmark.
Note that this assumes to have downloaded the benchmark data to `mrqa-few-shot` as specified in `data/datasets.ini`.

In general, datasets can be referenced by a name from [HF's datasets library](https://huggingface.co/datasets) or as specified in `data/datasets.ini`.

See `utils/args.py` for arguments or run `python run.py -h`.
Many arguments are provided by transformers (see https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.TrainingArguments for a list and description of these parameters).

Prediction can be run with e.g., 
```bash
python run.py qg \
    --output_dir models/qg-squad-16 \
    --load_checkpoint models/qg-squad-16 \
    --load_best_checkpoint \
    --transformer t5-v1-1-large \
    --prompt-method soft \
    --template-idx 22 \
    --lang en \
    --num_worker 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \
    --do_predict \
    --predict-dataset squad:train \
    --predict-dir gen_data/squad_train \
    --predict_chunking_mode token \
    --predict_context_stride 100 \
    --predict_context_size 450 \
    --bf16_full_eval \
    --answer-sampler ner
````
where the question generation model is loaded from `models/qg-squad-16` (using its best checkpoint) and documents for generation are taken from `squad:train`.

# QA Model Training

In general, training a QA model using Prompting is done using the `rc` argument of the `run.py` script.
For example, to train and evaluate a QA model on SQuAD, run
```bash
python run.py rc \
    --output_dir models/rc-squad \
    --transformer t5-v1-1-large \
    --prompt-method soft \
    --template-idx 13 \
    --lang en \
    --num_worker 5 \
    --do_train \
    --num_train_epochs 3 \
    --train-dataset st-squad:train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --train_chunking_mode token \
    --train_context_stride 100 \
    --train_context_size 450 \
    --do_eval \
    --eval-dataset st-squad:validation \
    --per_device_eval_batch_size 1 \
    --eval_chunking_mode token \
    --eval_context_stride 100 \
    --eval_context_size 450 \
    --bf16_full_eval \
    --answer-sampler ner
```
.

# Citation
```bibtex
@inproceedings{schmidt-etal-2024-prompting-based,
    title = "Prompting-based Synthetic Data Generation for Few-Shot Question Answering",
    author = "Schmidt, Maximilian  and
      Bartezzaghi, Andrea  and
      Vu, Ngoc Thang",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1153",
    pages = "13168--13178",
}

```