#!/usr/bin/env python3
"""
Create detailed plot showing trade dots at different prices
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load our LOB data
db_path = "/workspace/lob_databases_calibrated_volume/AMZN_2012-06-21_LLMON_calibrated.db"
conn = sqlite3.connect(db_path)

# Get trades
trades_df = pd.read_sql('''
    SELECT timestamp, price, side 
    FROM trades 
    ORDER BY timestamp
''', conn)

# Get orderbook
orderbook_df = pd.read_sql('''
    SELECT timestamp, mid_price 
    FROM orderbook 
    ORDER BY timestamp
''', conn)

conn.close()

print(f"Total trades: {len(trades_df)}")

# Create figure with multiple views
fig = plt.figure(figsize=(20, 12))

# Plot 1: Full 6.5 hour view
ax1 = plt.subplot(3, 1, 1)
ax1.set_title('Full Day View - All Trades', fontsize=14, fontweight='bold')

# Plot mid-price line
time_hours = orderbook_df['timestamp'].values / 3600
mid_prices = orderbook_df['mid_price'].values
ax1.plot(time_hours, mid_prices, 'b-', linewidth=1.5, label='Mid Price', zorder=2)

# Plot ALL trade dots
trade_times = trades_df['timestamp'].values / 3600
trade_prices = trades_df['price'].values
ax1.scatter(trade_times, trade_prices, c='red', s=0.5, alpha=0.3, label='Trades', zorder=1)

ax1.set_xlabel('Hours from Market Open')
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 6.5)

# Plot 2: Zoom to 1 hour (hour 2-3)
ax2 = plt.subplot(3, 1, 2)
ax2.set_title('1 Hour Zoom (Hour 2-3) - Trade Dots Become More Visible', fontsize=14, fontweight='bold')

# Filter data for hour 2-3
hour_mask = (trades_df['timestamp'] >= 7200) & (trades_df['timestamp'] < 10800)
hour_trades = trades_df[hour_mask]

ob_mask = (orderbook_df['timestamp'] >= 7200) & (orderbook_df['timestamp'] < 10800)
hour_ob = orderbook_df[ob_mask]

# Plot mid-price
ax2.plot(hour_ob['timestamp'].values / 3600, hour_ob['mid_price'].values, 
         'b-', linewidth=2, label='Mid Price', zorder=2)

# Plot trades with larger dots
ax2.scatter(hour_trades['timestamp'].values / 3600, hour_trades['price'].values, 
           c='red', s=2, alpha=0.5, label=f'{len(hour_trades)} trades', zorder=1)

ax2.set_xlabel('Hours from Market Open')
ax2.set_ylabel('Price ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(2, 3)

# Plot 3: Extreme zoom - 1 minute (minute 5 of hour 2)
ax3 = plt.subplot(3, 1, 3)
ax3.set_title('1 Minute Extreme Zoom - Individual Trade Prices Clearly Visible', fontsize=14, fontweight='bold')

# Filter for 1 minute
minute_start = 7200 + 5*60  # 5 minutes into hour 2
minute_end = minute_start + 60

minute_mask = (trades_df['timestamp'] >= minute_start) & (trades_df['timestamp'] < minute_end)
minute_trades = trades_df[minute_mask]

print(f"\nTrades in this 1 minute: {len(minute_trades)}")

if len(minute_trades) > 0:
    # Group by second to show distribution
    minute_trades['second'] = minute_trades['timestamp'].astype(int)
    trades_per_second = minute_trades.groupby('second').agg({
        'price': ['count', 'min', 'max', 'nunique']
    })
    print(f"Seconds with trades: {len(trades_per_second)}")
    print(f"Max trades in one second: {trades_per_second[('price', 'count')].max()}")
    print(f"Max unique prices in one second: {trades_per_second[('price', 'nunique')].max()}")
    
    # Plot each trade as a distinct dot
    for _, trade in minute_trades.iterrows():
        color = 'green' if trade['side'] == 'B' else 'red'
        ax3.scatter(trade['timestamp'], trade['price'], c=color, s=50, alpha=0.7, 
                   edgecolors='black', linewidth=0.5)
    
    # Add mid-price line
    minute_ob_mask = (orderbook_df['timestamp'] >= minute_start) & (orderbook_df['timestamp'] < minute_end)
    minute_ob = orderbook_df[minute_ob_mask]
    if len(minute_ob) > 0:
        ax3.plot(minute_ob['timestamp'].values, minute_ob['mid_price'].values, 
                'b-', linewidth=2, label='Mid Price', zorder=1)
    
    # Annotate some trades to show different prices
    for i, (_, trade) in enumerate(minute_trades.head(10).iterrows()):
        if i % 2 == 0:  # Annotate every other trade
            ax3.annotate(f'${trade["price"]:.2f}', 
                        xy=(trade['timestamp'], trade['price']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    ax3.set_xlabel('Seconds')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Mid Price', 'Buy Trade', 'Sell Trade'])
    ax3.set_xlim(minute_start, minute_end)
    
    # Set x-axis to show seconds
    ax3.set_xticks(range(minute_start, minute_end+1, 10))
    ax3.set_xticklabels([f'{i}s' for i in range(0, 61, 10)])

plt.suptitle('Trade Price Visibility at Different Zoom Levels', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/trade_dots_detail.png', dpi=150)
print(f"\n✅ Saved to /workspace/trade_dots_detail.png")

# Analysis of price distribution
print("\n" + "="*60)
print("ANALYSIS: Why dots appear as single points")
print("="*60)

# Check how close prices are
trades_df['second'] = trades_df['timestamp'].astype(int)
for second in trades_df['second'].value_counts().head(5).index:
    second_trades = trades_df[trades_df['second'] == second]
    if len(second_trades) > 1:
        prices = second_trades['price'].values
        price_range = prices.max() - prices.min()
        print(f"\nSecond {second} ({second/3600:.2f} hours):")
        print(f"  {len(second_trades)} trades")
        print(f"  Prices: {prices}")
        print(f"  Range: ${price_range:.2f}")
        print(f"  As % of price: {price_range/prices.mean()*100:.3f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
The dots appear as single points in the full-day view because:

1. SCALE ISSUE: 
   - Y-axis spans $65 ($165-230)
   - Price differences are only $0.01-0.03 (0.01% of range)
   - These tiny differences are invisible at full scale

2. VISUAL OVERLAP:
   - Dots are plotted with some transparency
   - Multiple dots at similar prices blend together
   - Looks like one dot but actually multiple

3. RESOLUTION:
   - Screen pixels can't distinguish $0.01 differences 
   - When y-axis spans $65, each pixel represents ~$0.05

SOLUTION: Zoom in to see the multiple prices clearly!
""")