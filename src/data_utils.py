# Функция для "чистки" текстов.
def clean_string(text):
    text = re.sub(r'(#\S+)|(@\S+)|(https?://\S+)|(www\.\S+)', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]|[^a-zA-Z0-9\s.,!?;:\-()"\'$%&*+=]', '', text)
    text = re.sub(r'([!?.,;:])\1\1+', r'\1', text)
    return text


# Создаем сырой датасет в формате .csv как по ТЗ.
with open('data/tweets.txt', 'r', encoding='utf-8') as f_in, \
        open('data/raw_dataset.csv', 'w', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['tweet'])
        
        for line in f_in:
            writer.writerow([line.lower().strip()])


# Проводим очистку твитов и создаем датасет в формате .csv как по ТЗ.
with open('data/raw_dataset.csv', 'r', encoding='utf-8') as start, \
     open('data/dataset_processed.csv', 'w', newline='', encoding='utf-8') as finish:
    
    reader = csv.reader(start)
    writer = csv.writer(finish)
    
    next(reader)  # пропускаем заголовок.
    writer.writerow(['tweet'])  # пишем новый заголовок.
    
    for row in reader:
        if not row:  # пустые строки пропускаем.
            continue
            
        cleaned_text = clean_string(row[0])
        if not cleaned_text:  # пропускаем, если очистка вернула пустоту.
            continue
            
        # Строка наполняется при наличии от 4 слов.
        words = cleaned_text.split()
        if len(words) > 3:
            writer.writerow([cleaned_text])


# Получаем список из очищенного датасета.
raw = pd.read_csv('data/dataset_processed.csv')
cleaned_texts = raw['tweet'].to_list()

print('Датасет sentiment140 загружен, для него определена предобработка (чистка\
 лишних символов, lowercase, удаление повторяющихся пробелов и т. д.).')

# Разбиение на тренировочную, валидационную и тестовую выборки.
train_texts, val_test_texts = train_test_split(cleaned_texts, test_size=0.2, random_state=42)
val_texts, test_texts = train_test_split(val_test_texts, test_size=0.5, random_state=42)

# Загружаем BERT токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Токенизируем данные и сохраняем их в .csv.
def tokenize_and_save(texts, tokenizer, output_file, mode = 'train_test', max_len=data_cfg['max_len']):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['context_tokens', 'target_token', 'status'])
        
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=True,
                                         truncation=True, max_length=max_len,
                                         padding=False)
            k = len(line) * 3 // 4
            context = token_ids[:k]
            target = token_ids[1:k+1]
                        
            # Паддинг до одинаковой длины.
            if len(context) < max_len:
                pad_len = max_len - len(context)
                context = context + [tokenizer.pad_token_id] * pad_len
            else:
                context = context[:max_len]

            writer.writerow([context, target, 0])

            # Для вычисления ROUGE дополнительно сохраняем целые токенизированные примеры.
            if mode == 'val':
                full_tokens = token_ids.copy()
                if len(full_tokens) < max_len:
                      full_tokens = full_tokens + [tokenizer.pad_token_id] * \
                        (max_len - len(full_tokens))
                else:
                    full_tokens = full_tokens[:max_len]
                    
                writer.writerow([full_tokens, tokenizer.pad_token_id, 1])


if flag1:
    tokenize_and_save(train_texts, tokenizer, 'data/train.csv')
    tokenize_and_save(val_texts, tokenizer, 'data/val.csv', mode='val')
    tokenize_and_save(test_texts, tokenizer, 'data/test.csv')

print('Датасет разбит на трейн, валидацию, тест.')
