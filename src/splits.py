def chronological_split(dataset, train_fraction):
    n = len(dataset)
    split_idx = int(n * train_fraction)
    train_ds = dataset[:split_idx]
    val_ds = dataset[split_idx:]
    return train_ds, val_ds
