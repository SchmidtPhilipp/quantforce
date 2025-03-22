from torch.utils.data import DataLoader

def get_loader(df, batch_size=32, window_size=30, shuffle=False):
    dataset = PortfolioDataset(df, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
