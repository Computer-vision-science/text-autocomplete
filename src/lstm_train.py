optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Основной цикл обучения.
n_epochs = 20

# Собираем данные об ошибках на обучающем и тестовом датасете.
train_stat, test_stat = [], []

# обучение с нуля
if not flag2:
    resume_epoch = 0

print('Написан код обучения модели.')


for epoch in range(n_epochs - resume_epoch):
    model.train()
    train_loss = 0
    for x_batch, y_batch in tqdm(train_loader, desc='Запущено обучение модели.\
 На протяжении обучения метрики выводятся в виде текста.'):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_batch, y_batch = x_batch.long(), y_batch.long()
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_stat.append(train_loss)
    val_loss, val_acc = evaluate_loss_acc(model, val_loader_loss)
    print(f"Epoch {epoch+1+resume_epoch} | Train Loss: {train_loss:.3f} |\
 Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

    # Теперь для подсчета ROUGE берем случайные элементы из валидационной выборки.
    random_indices = random.sample(range(len(val_dataset_rouge)), 100)
    rouge1_total, rouge2_total = 0, 0


    for i in tqdm(random_indices, desc='В коде обучения реализован замер качества модели:\
 функция потерь и метрика ROUGE на валидационном датасете.', leave=False):
        context_tokens, _ = val_dataset_rouge[i]

        # Преобразуем в список и убираем специальные токены.
        tokens = context_tokens.tolist()
        tokens = [t for t in tokens if t not in [0, 101, 102]]
        split_point = len(tokens) * 3 // 4
        context = tokens[:split_point]
        expected_tokens = tokens[split_point:]
                
        generated_text = model.several(context, len(expected_tokens))
        expected_text = tokenizer.decode(expected_tokens, skip_special_tokens=True)
                
        scores = rouge_ind(generated_text, expected_text)
        rouge1_total += scores['rouge1']
        rouge2_total += scores['rouge2']


    print(f"ROUGE-1: {rouge1_total/100:.4f}")
    print(f"ROUGE-2: {rouge2_total/100:.4f}")

    # Теперь подсчитаем loss, acc для тестовой выборки
    model.eval()
    test_loss = 0


    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_batch, y_batch = x_batch.long(), y_batch.long()
            logits = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            test_loss += loss.item()


    test_loss /= len(test_loader)
    test_stat.append(test_loss)
    print(f"Test Loss: {test_loss:.3f}")

    checkpoint = {
                  'epoch': epoch + 1+resume_epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'train_loss': train_loss,
                  'val_loss': val_loss,
                  'val_acc': val_acc,
                  'rouge1': rouge1_total/100,
                  'rouge2': rouge2_total/100,
                  'train_stat': train_stat,  # вся история тренировочных потерь
                  'test_stat': test_stat,     # вся история тестовых потерь
}

    torch.save(checkpoint, f'results/checkpoint_epoch_{epoch+1+resume_epoch}.pth')
    print('Сохранены веса обученной модели.')


    if len(train_stat) > 2:
        train_trend = train_stat[-1] > train_stat[-3]
        test_trend = test_stat[-1] > test_stat[-3]
        if train_trend and test_trend:
            print('Обучение прервано, ошибка не уменьшается')
            break
        print('В процессе обучения ошибки на обучающем и тестовом датасете \
 преимущественно падают (возможны небольшие отклонения от тренда на протяжении 1-3 эпох).')
