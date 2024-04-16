import numpy as np

def find_all_arbitrages(table, max_trades, start_commodity):
    num_commodities = len(table)
    profitable_paths = []

    # Recursive function to explore all possible trade paths
    def trade(current_commodity, current_profit, num_trades, path):
        if num_trades == max_trades:
            # Check if we can close the cycle back to 'S' with a profit
            if current_commodity == start_commodity:
                final_profit = current_profit
                if final_profit > 1:  # Only consider profitable trades
                    profitable_paths.append((path, final_profit))
            return
        
        for next_commodity in range(num_commodities):
            if next_commodity != current_commodity:  # No self-loop
                new_profit = current_profit * table[current_commodity][next_commodity]
                trade(next_commodity, new_profit, num_trades + 1, path + [next_commodity])

    # Start the arbitrage cycle from 'S'
    trade(start_commodity, 1, 0, [start_commodity])
    
    return profitable_paths

# Define the exchange rate table
exchange_rates = np.array([
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.3, 1, 0.46],
    [1.41, 0.61, 2.08, 1]
])

max_trades = 5
start_commodity = 3  # Index of 'S'

# Find all arbitrage opportunities starting and ending with 'S'
profitable_paths = find_all_arbitrages(exchange_rates, max_trades, start_commodity)

print("Profitable trading paths starting and ending with 'S':")
for path, profit in profitable_paths:
    final_units = 2000000 * profit
    print(f"Path: {path} -> Profit Multiplier: {profit:.4f} -> Final Units of 'S': {final_units:.2f}, thanks Alex")
