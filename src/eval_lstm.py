# Функция замера лосса и accuracy

rouge_metric = evaluate.load("rouge")


def evaluate_loss_acc(model, loader, pad_token_id=tokenizer.pad_token_id):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Validation", leave=False):
            x_batch = x_batch.to(device).long()
            y_batch = y_batch.to(device).long()
            x_output = model(x_batch)

            if hasattr(x_output, 'logits'):  # Для GPT-2 (HuggingFace модели)
                logits = x_output.logits
            else:  # Для LSTM и других моделей, которые возвращают тензор
                logits = x_output
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            sum_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            mask = (y_batch != pad_token_id)
            correct += ((preds == y_batch) & mask).sum().item()
            total += mask.sum().item()
    return sum_loss / len(loader), correct / total


# Функция замера ROUGE
def rouge_ind(X, Y):
    results = rouge_metric.compute(predictions=[X], references=[Y])
    return results
