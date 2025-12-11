import datetime
import json
import os
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster
from sklearn.preprocessing import RobustScaler

def ensure_mapping_exists(filename):
    """Створює JSON файл, якщо його немає."""
    if not os.path.exists(filename):
        print(f"Створення {filename}...")
        companies = {
            "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOG": "Google",
            "IBM": "IBM", "INTC": "Intel", "BA": "Boeing", "CAT": "Caterpillar",
            "CVX": "Chevron", "XOM": "Exxon", "KO": "Coca-Cola", "PEP": "Pepsi",
            "JPM": "JPMorgan Chase", "C": "Citigroup", "WFC": "Wells Fargo"
        }
        with open(filename, 'w') as f:
            json.dump(companies, f)

def main():
    input_file = 'company_symbol_mapping.json'
    ensure_mapping_exists(input_file)

    with open(input_file, 'r') as f:
        company_symbols_map = json.load(f)
    symbols = list(company_symbols_map.keys())

    start_date = "2020-01-01"
    end_date = "2023-01-01"
    print(f"Завантаження даних для {len(symbols)} компаній...")
    
    data = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=True)

    try:
        if 'Open' in data.columns and isinstance(data.columns, np.ndarray):
             opening_quotes = data['Open']
             closing_quotes = data['Close']
        else:
             opening_quotes = data['Open'] if 'Open' in data else data.xs('Open', level=0, axis=1)
             closing_quotes = data['Close'] if 'Close' in data else data.xs('Close', level=0, axis=1)
    except Exception as e:
        print(f"Помилка структури даних: {e}")
        return

    quotes_diff = closing_quotes - opening_quotes

    quotes_diff.dropna(axis=1, how='all', inplace=True)
    quotes_diff.dropna(axis=0, how='any', inplace=True)

    X = quotes_diff.copy().values
    
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    edge_model = covariance.GraphicalLassoCV(cv=5, assume_centered=True)
    edge_model.fit(X)

    median_val = np.median(edge_model.covariance_)
    af_model = cluster.AffinityPropagation(preference=median_val, random_state=42)
    af_model.fit(edge_model.covariance_)

    labels = af_model.labels_
    num_labels = labels.max()
    
    valid_symbols = quotes_diff.columns
    names = np.array([company_symbols_map.get(s, s) for s in valid_symbols])

    print("\n--- Результати кластеризації компаній ---")
    for i in range(num_labels + 1):
        cluster_members = names[labels == i]
        print(f"Cluster {i+1} ==> {', '.join(cluster_members)}")

if __name__ == "__main__":
    main()