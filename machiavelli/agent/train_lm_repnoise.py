import numpy as np
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(os.path.join(project_root, "machiavelli"))

from transformers import AutoModelForTokenClassification, AutoModelForCausalLM

from representation_noising_xpo.representation_noising.trainers.repnoise import train_repnoise_model
from representation_noising_xpo.representation_noising.datasets import construct_beavertails_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
#model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
#model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
#model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token

trainds, testds = construct_beavertails_dataset(tokenizer=tokenizer, attack_type="sft", train_implicit_defence=True)

train_repnoise_model(model, tokenizer, trainds, testds)
