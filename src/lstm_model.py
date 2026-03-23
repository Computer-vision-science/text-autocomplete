class NNLSTM(nn.Module):
    def __init__(
                 self, vocab_size, hidden_dim=lstm_cfg['hidden_dim'],\
                 dropout=lstm_cfg['dropout'], num_layers=lstm_cfg['num_layers']
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x.long())
        drop = self.dropout1(emb)
        out, _ = self.rnn(drop)
        linear_out = self.fc(out)
        return linear_out

    # Генерация заданного количества слов.
    def several(self, x, n):
        self.eval()
        device = next(self.parameters()).device
        x = torch.tensor([x], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(n):
                logits = self.forward(x)
                probs = logits[0, -1, :].softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

        result_text = tokenizer.decode(x[0], skip_special_tokens=True)
        return result_text
    
    
print('Реализован класс нейронной сети с LSTM в основе и метод forward для \
предсказания следующего токена.')
print('У модели реализован метод для предсказания нескольких следующих токенов.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NNLSTM(vocab_size=tokenizer.vocab_size)
model = model.to(device)

print('В переменной model лежит объект модели.')
print('Код этапа 2 рабочий, ошибок не вызывает.')
