class PortfolioDataset:
    def __init__(self, df, window_size=30):
        self.df = df
        self.window_size = window_size
        self.length = len(df) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = self.df.iloc[idx:idx+self.window_size].values
        target = self.df.iloc[idx+self.window_size].values
        return obs, target
