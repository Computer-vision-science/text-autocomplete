# Конвертация в формат .npz
def convert_csv_to_numpy(csv_file, output_file, status_filter=None, max_len=data_cfg['max_len']
):
    df = pd.read_csv(csv_file, usecols=['context_tokens', 'target_token', 'status'])
    
    if status_filter is not None:
        df = df[df['status'] == status_filter]
        df = df.reset_index(drop=True)

    n = len(df)
    contexts = np.zeros((n, max_len), dtype=np.int32)
    targets = np.zeros((n, max_len), dtype=np.int32)
    
    for i, row in tqdm(df.iterrows(), total=n):
        c = str(row['context_tokens']).strip('[]')
        if c:
            ctx = [int(x) for x in c.split(',') if x.strip()]
            # ctx уже должен быть длины max_len из CSV.
            contexts[i] = ctx[:max_len]  # просто копируем.
            
        t = str(row['target_token']).strip('[]')
        if t and t != '0':
            target_list = [int(x) for x in t.split(',') if x.strip()]
            targets[i, :len(target_list)] = target_list[:max_len]
    
    np.savez_compressed(output_file, contexts=contexts, targets=targets)


class NumpyDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.contexts = data['contexts']
        self.targets = data['targets']
        print(f"Загружено {len(self.targets)} примеров из {npz_file}")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


if flag1:
    convert_csv_to_numpy('data/train.csv', 'data/train.npz')
    convert_csv_to_numpy('data/val.csv', 'data/val_loss.npz', status_filter=0)
    convert_csv_to_numpy('data/val.csv', 'data/val_rouge.npz', status_filter=1)
    convert_csv_to_numpy('data/test.csv', 'data/test.npz')


def collate_fn(batch):
    contexts, targets = zip(*batch)
    return (torch.from_numpy(np.stack(contexts)),
            torch.from_numpy(np.stack(targets)))


train_dataset = NumpyDataset('data/train.npz')
val_dataset_loss = NumpyDataset('data/val_loss.npz')    # status=0
val_dataset_rouge = NumpyDataset('data/val_rouge.npz')  # status=1
test_dataset = NumpyDataset('data/test.npz')

print("Тренировочный, валидационный и тестовые датасеты помещены в torch.Dataset'ы.")

# Даталоадеры
train_loader = DataLoader(
    train_dataset, batch_size=lstm_cfg['batch_size'], shuffle=True, collate_fn=collate_fn, pin_memory=True
    )
val_loader_loss = DataLoader(
    val_dataset_loss, batch_size=lstm_cfg['batch_size'], collate_fn=collate_fn, pin_memory=True
    )
val_loader_rouge = DataLoader(
    val_dataset_rouge,  batch_size=lstm_cfg['batch_size'], collate_fn=collate_fn, pin_memory=True
    )
test_loader = DataLoader(
    test_dataset, batch_size=lstm_cfg['batch_size'], collate_fn=collate_fn, pin_memory=True
    )

print('Под каждый torch.Dataset создан соответствующий Dataloader.')
print('Код этапа 1 рабочий, ошибок не вызывает.')
