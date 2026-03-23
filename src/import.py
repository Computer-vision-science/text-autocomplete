from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizerFast, pipeline
import csv, evaluate, random, re, torch, yaml
import pandas as pd
import torch.nn as nn
import numpy as np


# Загружаем конфигурацию
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Достаем параметры
lstm_cfg = config['lstm']
gpt2_cfg = config['gpt2']
data_cfg = config['data']

print('Создан репозиторий на GitHub github.com/Computer-vision-science/text-autocomplete,\
 файловая структура понятна и схожа со структурой,рекомендованной в инструкции к проекту. \
 Для навигации и понимания, что происходит, к проекту написано описание в файле README.md.')
print('Для навигации и понимания, что происходит, к проекту написано описание в файле README.md.')
