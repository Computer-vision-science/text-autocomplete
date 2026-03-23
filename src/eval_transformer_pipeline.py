# Загрузка токенизатора и модели.
model_name = gpt2_cfg['model_name']
tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model_name)

print('Скачана модель предобученного трансформера distilgpt2.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gpt2 = model_gpt2.to(device)

# Создаём pipeline для генерации.
generator = pipeline(
    task="text-generation",
    model=model_gpt2,
    tokenizer=tokenizer_gpt2,
    device=0  # -1 = CPU; 0 = первый GPU.
)

print('Написан рабочий код генерации текстов.')

val_loss_gpt2, val_acc_gpt2 = evaluate_loss_acc(model_gpt2, val_loader_loss)

print(f"Val Loss: {val_loss_gpt2:.3f} | Val Accuracy: {val_acc_gpt2:.2%}")
print('Написан код валидации модели.')

# Теперь для подсчета ROUGE берем все элементы из валидационной выборки.

rouge1_total, rouge2_total = 0, 0

random_indices = random.sample(range(len(val_dataset_rouge)), min(1000, len(val_dataset_rouge)))

for i in tqdm(random_indices, desc='Сгенерированы автодополнения для текстов,\
 замерено значение метрики ROUGE, выведены примеры предсказаний.', leave=False):
    context_tokens, _ = val_dataset_rouge[i]

    # Преобразуем в список и убираем специальные токены.
    tokens = context_tokens.tolist()
    tokens = [t for t in tokens if t not in [0, 101, 102]]
    split_point = len(tokens) * 3 // 4
    context = tokens[:split_point]
    context = tokenizer_gpt2.decode(context)
    target = tokens[split_point:]
    target = tokenizer_gpt2.decode(target)
    max_new_tokens = len(tokens) - split_point    

    out = generator(context,
    max_new_tokens=max_new_tokens,    
    num_return_sequences=1,
    do_sample=True,    
    top_p=gpt2_cfg["top_p"],      
    temperature=gpt2_cfg['temperature']
)
    full_text = out[0]["generated_text"]
    generated_part = full_text[len(context):].strip()
                
    scores = rouge_ind(generated_part, target)
    rouge1_total += scores['rouge1']
    rouge2_total += scores['rouge2']

rouge1_gpt2 = rouge1_total / 1000
rouge2_gpt2 = rouge2_total / 1000

print(f"Average ROUGE-1: {rouge1_gpt2:.4f}")
print(f"Average ROUGE-2: {rouge2_gpt2:.4f}")
