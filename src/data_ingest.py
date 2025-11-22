import pandas as pd

transactions = pd.read_csv("data/schwab_transactions.csv", skip_blank_lines=True, header=0)


print(transactions.head())

