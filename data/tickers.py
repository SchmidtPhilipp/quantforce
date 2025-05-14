import pandas as pd


NASDAQ100 = ["KLAC", "ORLY", "HON", "AMZN", "GFS", "ASML", "INTU", "MCHP", "BIIB", "IDXX", "MDB", "COST", "FAST", "TXN", "WBD", "ISRG", "CRWD", "PCAR", "META", "MDLZ", "LRCX", "NVDA", "MNST", "SNPS", "NXPI", "ADBE", "TTD", "FANG", "BKNG", "KDP", "TMUS", "AMD", "ADI", "PYPL", "CEG", "LULU", "INTC", "AZN", "EXC", "MELI", "BKR", "CDNS", "TEAM", "LIN", "ADSK", "CSCO", "QCOM", "CTSH", "CSGP", "AVGO", "PLTR", "ADP", "AAPL", "WDAY", "CSX", "NFLX", "REGN", "ZS", "MSTR", "CHTR", "PANW", "ABNB", "GOOGL", "ODFL", "CPRT", "APP", "ANSS", "SBUX", "FTNT", "VRTX", "PDD", "ARM", "TSLA", "AXON", "VRSK", "ROP", "TTWO", "PAYX", "CDW", "MSFT", "CTAS", "EA", "AMAT", "DXCM", "MRVL", "MU", "ROST", "PEP", "MAR"]
DOWJONES = ["DIS", "VZ", "HON", "IBM", "JNJ", "AMZN", "CSCO", "GS", "MRK", "SHW", "CAT", "TRV", "MCD", "AAPL", "KO", "UNH", "NKE", "BA", "CVX", "MSFT", "NVDA", "PG", "WMT", "V"]
SNP_500 = ["WFC", "SBAC", "KLAC", "HCA", "KIM", "MAS", "DHI", "ES", "DOW", "BDX", "AMZN", "WAT", "DOV", "JKHY", "NVR", "LVS", "ECL", "EXPE", "LLY", "CI", "PSX", "MCHP", "FDS", "DECK", "KO", "BIIB", "HSIC", "IDXX", "HSY", "GDDY", "DVN", "MPC", "TYL", "HUBB", "TXN", "WELL", "DELL", "COST", "FAST", "LUV", "BBY", "VTRS", "PCG", "STT", "PCAR", "META", "SRE", "UHS", "FIS", "MDLZ", "RL", "SJM", "ALLE", "NRG", "KKR", "YUM", "IQV", "CNC", "BG", "CMG", "STX", "JNJ", "V", "LYB", "ADBE", "IPG", "J", "BKNG", "TRMB", "TFC", "MMC", "ROL", "MRK", "UAL", "NOW", "PEG", "IT", "WYNN", "IP", "PTC", "TT", "VICI", "AZO", "KDP", "UPS", "EW", "MCD", "UNH", "HAS", "ADI", "STE", "PYPL", "BR", "HWM", "AVY", "NWSA", "NDSN", "MCK", "APA", "CVX", "KEYS", "RTX", "ERIE", "GEN", "RF", "SMCI", "EG", "TKO", "CB", "GLW", "CPB", "KMI", "EXR", "ABBV", "WBA", "ICE", "LNT", "CDNS", "BMY", "DRI", "EVRG", "IBM", "LIN", "OMC", "LH", "REG", "PPL", "ITW", "CTSH", "NEE", "PFG", "WSM", "SLB", "DFS", "TSCO", "GWW", "FFIV", "ADP", "AAPL", "CSX", "BSX", "EQR", "EOG", "EFX", "L", "NFLX", "CCL", "REGN", "PHM", "AIG", "DAY", "MHK", "UDR", "COO", "TXT", "ENPH", "PNW", "KEY", "SO", "ALGN", "CINF", "FOXA", "MRNA", "ABNB", "AKAM", "MSI", "MTCH", "CPRT", "URI", "EMR", "PPG", "SBUX", "FSLR", "NI", "FTNT", "IVZ", "VRTX", "GS", "SWKS", "HPE", "GIS", "ELV", "TDG", "EBAY", "WAB", "CNP", "PRU", "F", "PAYC", "ED", "NCLH", "NEM", "WM", "AXON", "CFG", "AMT", "EXPD", "WRB", "VRSK", "APO", "TGT", "GEV", "KMX", "ARE", "ROP", "EL", "PLD", "LMT", "AOS", "NUE", "MOH", "CPAY", "PAYX", "CDW", "MSFT", "SCHW", "CTAS", "PG", "DAL", "BEN", "PM", "AON", "XYL", "BAX", "FDX", "TECH", "CAG", "SYF", "BKR", "TPR", "DIS", "ORLY", "HPQ", "HON", "CVS", "EIX", "MCO", "TAP", "HUM", "USB", "SHW", "INTU", "ORCL", "TRV", "ABT", "HAL", "HST", "CRM", "TMO", "UBER", "K", "FOX", "CBRE", "NKE", "JNPR", "MPWR", "WMT", "GPN", "SOLV", "WBD", "DOC", "DGX", "VLTO", "ISRG", "CRL", "CRWD", "WY", "AFL", "ALB", "DE", "LRCX", "NVDA", "APD", "MNST", "FRT", "PNR", "ACN", "SNPS", "RCL", "OTIS", "TRGP", "CL", "MLM", "NXPI", "O", "XEL", "VZ", "CPT", "GNRC", "FANG", "ALL", "PODD", "TEL", "NTAP", "OXY", "ATO", "LYV", "VRSN", "APTV", "HOLX", "CBOE", "LOW", "MOS", "PGR", "OKE", "INVH", "TMUS", "AVB", "AMD", "KR", "CZR", "ETR", "FICO", "ANET", "CF", "MS", "CEG", "DLR", "LULU", "DPZ", "JCI", "IEX", "BAC", "FTV", "CLX", "INTC", "AWK", "CHRW", "EXC", "HRL", "FCX", "AEE", "CMI", "GD", "AES", "RSG", "NWS", "TROW", "EXE", "WDC", "SPG", "VMC", "ADSK", "CSCO", "GRMN", "QCOM", "TJX", "CSGP", "GPC", "AVGO", "HIG", "EQT", "PLTR", "C", "BLDR", "POOL", "WDAY", "ETN", "COR", "FE", "BX", "GE", "BRO", "PSA", "TSN", "LII", "SYK", "RMD", "MKTX", "CTVA", "ULTA", "MGM", "MO", "BLK", "MTD", "VLO", "PANW", "CHTR", "COP", "CARR", "CMS", "VTR", "RVTY", "GOOGL", "ODFL", "CHD", "XOM", "DTE", "MAA", "ADM", "LEN", "ACGL", "HII", "RJF", "ANSS", "PH", "STLD", "IRM", "MTB", "DVA", "CTRA", "EPAM", "CAT", "NSC", "UNP", "TER", "NOC", "MKC", "FITB", "MSCI", "AIZ", "T", "HBAN", "LKQ", "TSLA", "LDOS", "PWR", "SYY", "LW", "BXP", "APH", "BA", "NDAQ", "TTWO", "IR", "BALL", "WMB", "EA", "AJG", "AMAT", "D", "KVUE", "MDT", "PKG", "DHR", "MA", "DXCM", "ESS", "WEC", "MU", "ROST", "EQIX", "STZ", "TDY", "PEP", "BK", "MAR"]



def getNasdaq100():
    """
    Downloads the NASDAQ-100 data from Wikipedia and manages it.
    """
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100' 
    ticker_table = pd.read_html(url)[4]

    tickers = ticker_table["Ticker"].values.tolist()
    return tickers


def getDowJones():
    """
    Downloads the Dow Jones data from Wikipedia and manages it.
    """
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    ticker_table =pd.read_html(url)[2]
    
    tickers = ticker_table["Symbol"].values.tolist()
    return tickers

def getSAndP500():
    """
    Downloads the S&P 500 data from Wikipedia and manages it.
    """

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    ticker_table = pd.read_html(url)[0]
    
    tickers = ticker_table["Symbol"].values.tolist()
    return tickers


if __name__ == "__main__":
    
    nasdaq100_tickers = getNasdaq100()
    print("NASDAQ-100 Tickers:", nasdaq100_tickers)
    dow_jones_tickers = getDowJones()
    print("Dow Jones Tickers:", dow_jones_tickers)
    sp500_tickers = getSAndP500()
    print("S&P 500 Tickers:", sp500_tickers)