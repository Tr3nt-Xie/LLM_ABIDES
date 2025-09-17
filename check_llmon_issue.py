#!/usr/bin/env python3
"""
Diagnose why LLMON appears to have only 3 hours of data
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check all three databases
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for idx, condition in enumerate(['LLMON', 'LLMOFF', 'Baseline']):
    db_path = f'/workspace/lob_databases_calibrated/AMZN_2012-06-21_{condition}_calibrated.db'
    conn = sqlite3.connect(db_path)
    
    # Load orderbook data
    ob_df = pd.read_sql('SELECT timestamp, mid_price FROM orderbook', conn)
    conn.close()
    
    # Convert to hours
    time_hours = ob_df['timestamp'].values / 3600
    prices = ob_df['mid_price'].values
    
    # Plot
    ax = axes[idx]
    ax.plot(time_hours, prices, linewidth=1)
    ax.set_title(f'{condition} - Full Data')
    ax.set_xlabel('Hours from market open')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7)
    
    # Add vertical lines at each hour
    for hour in range(1, 7):
        ax.axvline(x=hour, color='gray', linestyle='--', alpha=0.2)
    
    # Print diagnostics
    print(f'\n{condition}:')
    print(f'  Time range: {ob_df["timestamp"].min():.1f} - {ob_df["timestamp"].max():.1f} seconds')
    print(f'  Duration: {ob_df["timestamp"].max()/3600:.2f} hours')
    print(f'  Number of snapshots: {len(ob_df)}')
    print(f'  Price range: ${prices.min():.2f} - ${prices.max():.2f}')
    print(f'  Final price: ${prices[-1]:.2f}')
    print(f'  Price change: {(prices[-1]/prices[0] - 1)*100:.2f}%')
    
    # Check for sudden drops
    price_changes = np.diff(prices)
    max_drop_idx = np.argmin(price_changes)
    max_drop = price_changes[max_drop_idx]
    if abs(max_drop) > 2:  # More than $2 drop
        print(f'  WARNING: Large drop of ${max_drop:.2f} at hour {time_hours[max_drop_idx]:.2f}')
    
    # Check if price drops below certain threshold
    if np.any(prices < 216):
        drop_time = time_hours[prices < 216][0]
        print(f'  WARNING: Price drops below $216 at hour {drop_time:.2f}')

plt.tight_layout()
plt.savefig('/workspace/llmon_diagnostic.png', dpi=150)
print('\n✅ Diagnostic plot saved to /workspace/llmon_diagnostic.png')

# Now check specifically what happens around hour 3
print('\n' + '='*60)
print('DETAILED ANALYSIS AROUND HOUR 3')
print('='*60)

for condition in ['LLMON', 'LLMOFF', 'Baseline']:
    db_path = f'/workspace/lob_databases_calibrated/AMZN_2012-06-21_{condition}_calibrated.db'
    conn = sqlite3.connect(db_path)
    
    ob_df = pd.read_sql('SELECT timestamp, mid_price FROM orderbook', conn)
    conn.close()
    
    # Look at data around 3 hours (10800 seconds)
    around_3h = ob_df[(ob_df['timestamp'] >= 10000) & (ob_df['timestamp'] <= 11600)]
    
    print(f'\n{condition} around hour 3:')
    print(f'  Price at 2.8h: ${around_3h.iloc[0]["mid_price"]:.2f}')
    print(f'  Price at 3.0h: ${around_3h.iloc[len(around_3h)//2]["mid_price"]:.2f}')
    print(f'  Price at 3.2h: ${around_3h.iloc[-1]["mid_price"]:.2f}')
    
    # Calculate price drop
    price_drop = around_3h['mid_price'].min() - around_3h['mid_price'].iloc[0]
    print(f'  Max drop in this period: ${price_drop:.2f}')

print('\n' + '='*60)
print('HYPOTHESIS')
print('='*60)
print("""
The issue is likely that LLMON has a sharp price drop around hour 3
due to the concentration of negative news events. This causes:

1. The y-axis scale to expand to show the full price range
2. The later data to appear compressed or cut off
3. Visual appearance of only 3 hours of data

This is because we have 4 negative news events clustered together,
and LLMON overreacts to them, causing a cascade effect.
""")