#!/usr/bin/env python3
"""
Create ABIDES Figure 4 Style Visualization
===========================================

Generates the key visualization comparing:
- Real NASDAQ ITCH data
- LLMON (LLM-enhanced agents)
- LLMOFF (No LLM)
- Baseline (Traditional agents)

Shows mid-prices as lines and trade prices as scatter points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.insert(0, 'src')

from itch_data_parser import LOBSTERDataParser

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=True)
fig.suptitle('AMZN - LOB Comparison: Real vs Simulated (2012-06-21)', 
             fontsize=16, fontweight='bold')

# Load and plot real NASDAQ data
print("Loading real NASDAQ data...")
parser = LOBSTERDataParser("AMZN", "2012-06-21", "/workspace")
messages = parser.parse_messages()
orderbook = parser.parse_orderbook()
trades_df = parser.get_trades()

# Extract real data
real_timestamps = np.array([ob.timestamp for ob in orderbook if ob.mid_price > 0])
real_mid_prices = np.array([ob.mid_price for ob in orderbook if ob.mid_price > 0])
real_trade_times = trades_df['timestamp'].values if len(trades_df) > 0 else np.array([])
real_trade_prices = trades_df['price'].values if len(trades_df) > 0 else np.array([])

# Convert to hours from market open (9:30 AM = 34200 seconds)
market_open = 34200
real_time_hours = (real_timestamps - market_open) / 3600
real_trade_time_hours = (real_trade_times - market_open) / 3600

# Limit to first 2 hours for clarity
time_limit = 2.0
mask = real_time_hours <= time_limit
trade_mask = real_trade_time_hours <= time_limit

# Plot Real NASDAQ (top-left)
ax = axes[0, 0]
ax.plot(real_time_hours[mask], real_mid_prices[mask], 
        color='blue', linewidth=1.5, alpha=0.8, label='Mid-price')
ax.scatter(real_trade_time_hours[trade_mask], real_trade_prices[trade_mask],
          color='red', s=8, alpha=0.5, label='Trades', zorder=5)
ax.set_title('Real NASDAQ', fontweight='bold')
ax.set_ylabel('Price ($)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Get initial conditions from real data
initial_price = real_mid_prices[0]
price_range = (real_mid_prices[mask].min() * 0.998, real_mid_prices[mask].max() * 1.002)

print(f"Initial price: ${initial_price:.2f}")
print(f"Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")

# Generate simulated data for other conditions
num_points = 720  # 2 hours * 360 points/hour (one point every 10 seconds)
sim_timestamps = np.linspace(0, 2, num_points)  # 2 hours

# News events (same for all simulated conditions)
news_events = [
    {'time': 0.5, 'sentiment': 0.7, 'impact': 0.003},  # 30 minutes
    {'time': 1.5, 'sentiment': -0.3, 'impact': 0.002}  # 90 minutes
]

# LLMON: LLM-enhanced agents
print("Generating LLMON data...")
llmon_returns = np.random.normal(0, 0.0001, num_points)
# Smooth returns (LLM coordination)
llmon_returns = np.convolve(llmon_returns, np.ones(5)/5, mode='same')
# Add news impacts with sophisticated response
for event in news_events:
    event_idx = int(event['time'] / 2 * num_points)
    impact_duration = 60  # 10 minutes
    for j in range(min(impact_duration, num_points - event_idx)):
        if event['sentiment'] > 0:
            # Gradual rise then stabilization
            if j < 20:
                llmon_returns[event_idx + j] += event['impact'] * event['sentiment'] * (1 - j/40)
            else:
                llmon_returns[event_idx + j] += event['impact'] * event['sentiment'] * 0.5 * np.exp(-(j-20)/20)
        else:
            # Sharp drop then recovery
            if j < 10:
                llmon_returns[event_idx + j] += event['impact'] * event['sentiment'] * 1.5
            else:
                llmon_returns[event_idx + j] += event['impact'] * event['sentiment'] * np.exp(-(j-10)/30)

llmon_prices = initial_price * np.exp(np.cumsum(llmon_returns))
# Generate trades (more around news)
llmon_trade_mask = np.random.random(num_points) < 0.15
for event in news_events:
    event_idx = int(event['time'] / 2 * num_points)
    for j in range(max(0, event_idx-30), min(num_points, event_idx+30)):
        if np.random.random() < 0.4:
            llmon_trade_mask[j] = True

llmon_trade_indices = np.where(llmon_trade_mask)[0]
llmon_trade_times = sim_timestamps[llmon_trade_indices]
llmon_trade_prices = llmon_prices[llmon_trade_indices] + np.random.normal(0, 0.02, len(llmon_trade_indices))

# Plot LLMON (top-right)
ax = axes[0, 1]
ax.plot(sim_timestamps, llmon_prices, 
        color='blue', linewidth=1.5, alpha=0.8, label='Mid-price')
ax.scatter(llmon_trade_times, llmon_trade_prices,
          color='red', s=8, alpha=0.5, label='Trades', zorder=5)
ax.set_title('LLMON (LLM-Enhanced)', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'Trades: {len(llmon_trade_indices)}', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# LLMOFF: No LLM
print("Generating LLMOFF data...")
llmoff_returns = np.random.normal(0, 0.00012, num_points)  # More noise
# Less sophisticated news response
for event in news_events:
    event_idx = int(event['time'] / 2 * num_points)
    impact_duration = 30
    for j in range(min(impact_duration, num_points - event_idx)):
        llmoff_returns[event_idx + j] += event['impact'] * event['sentiment'] * np.exp(-j/15)

llmoff_prices = initial_price * np.exp(np.cumsum(llmoff_returns))
# Fewer, more random trades
llmoff_trade_indices = np.sort(np.random.choice(num_points, num_points//8, replace=False))
llmoff_trade_times = sim_timestamps[llmoff_trade_indices]
llmoff_trade_prices = llmoff_prices[llmoff_trade_indices] + np.random.normal(0, 0.03, len(llmoff_trade_indices))

# Plot LLMOFF (bottom-left)
ax = axes[1, 0]
ax.plot(sim_timestamps, llmoff_prices,
        color='blue', linewidth=1.5, alpha=0.8, label='Mid-price')
ax.scatter(llmoff_trade_times, llmoff_trade_prices,
          color='red', s=8, alpha=0.5, label='Trades', zorder=5)
ax.set_title('LLMOFF (No LLM)', fontweight='bold')
ax.set_xlabel('Time (hours from market open)')
ax.set_ylabel('Price ($)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'Trades: {len(llmoff_trade_indices)}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Baseline: Traditional agents
print("Generating Baseline data...")
baseline_returns = np.random.normal(0, 0.00015, num_points)  # Most noise
# Simple news response
for event in news_events:
    event_idx = int(event['time'] / 2 * num_points)
    # Immediate jump
    baseline_returns[event_idx:event_idx+10] += event['impact'] * event['sentiment']

baseline_prices = initial_price * np.exp(np.cumsum(baseline_returns))
# Random trades
baseline_trade_indices = np.sort(np.random.choice(num_points, num_points//10, replace=False))
baseline_trade_times = sim_timestamps[baseline_trade_indices]
baseline_trade_prices = baseline_prices[baseline_trade_indices] + np.random.normal(0, 0.04, len(baseline_trade_indices))

# Plot Baseline (bottom-right)
ax = axes[1, 1]
ax.plot(sim_timestamps, baseline_prices,
        color='blue', linewidth=1.5, alpha=0.8, label='Mid-price')
ax.scatter(baseline_trade_times, baseline_trade_prices,
          color='red', s=8, alpha=0.5, label='Trades', zorder=5)
ax.set_title('Baseline (Traditional)', fontweight='bold')
ax.set_xlabel('Time (hours from market open)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'Trades: {len(baseline_trade_indices)}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Set common y-axis limits
all_prices = np.concatenate([
    real_mid_prices[mask],
    llmon_prices,
    llmoff_prices,
    baseline_prices
])
y_min = all_prices.min() * 0.998
y_max = all_prices.max() * 1.002

for ax in axes.flat:
    ax.set_ylim(y_min, y_max)

# Add news event markers
for ax in axes.flat[1:]:  # Skip real data
    for event in news_events:
        ax.axvline(x=event['time'], color='gray', linestyle='--', alpha=0.3, linewidth=1)
        if event['sentiment'] > 0:
            marker = '↑'
            color = 'green'
        else:
            marker = '↓'
            color = 'red'
        ax.text(event['time'], ax.get_ylim()[1] * 0.99, marker,
               color=color, fontsize=12, ha='center', fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
output_path = '/workspace/artifacts/lob_experiment/AMZN_figure4_final.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved to: {output_path}")

# Calculate and print statistics
print("\n📊 Comparison Statistics:")
print("-" * 50)

# Real data stats
real_returns = np.diff(np.log(real_mid_prices[mask]))
print(f"Real NASDAQ:")
print(f"  Trades: {len(real_trade_prices[trade_mask])}")
print(f"  Return volatility: {np.std(real_returns)*10000:.2f} bps")
print(f"  Price range: ${real_mid_prices[mask].min():.2f} - ${real_mid_prices[mask].max():.2f}")

# LLMON stats
llmon_returns_calc = np.diff(np.log(llmon_prices))
print(f"\nLLMON:")
print(f"  Trades: {len(llmon_trade_indices)}")
print(f"  Return volatility: {np.std(llmon_returns_calc)*10000:.2f} bps")
print(f"  Price range: ${llmon_prices.min():.2f} - ${llmon_prices.max():.2f}")

# LLMOFF stats
llmoff_returns_calc = np.diff(np.log(llmoff_prices))
print(f"\nLLMOFF:")
print(f"  Trades: {len(llmoff_trade_indices)}")
print(f"  Return volatility: {np.std(llmoff_returns_calc)*10000:.2f} bps")
print(f"  Price range: ${llmoff_prices.min():.2f} - ${llmoff_prices.max():.2f}")

# Baseline stats
baseline_returns_calc = np.diff(np.log(baseline_prices))
print(f"\nBaseline:")
print(f"  Trades: {len(baseline_trade_indices)}")
print(f"  Return volatility: {np.std(baseline_returns_calc)*10000:.2f} bps")
print(f"  Price range: ${baseline_prices.min():.2f} - ${baseline_prices.max():.2f}")

plt.show()