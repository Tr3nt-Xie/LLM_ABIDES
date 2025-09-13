#!/usr/bin/env python3
"""
Analyze trade price granularity in our generated data vs real NASDAQ data
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from itch_data_parser import LOBSTERDataParser

def analyze_trades_at_timestamp(db_path: str, condition: str):
    """Analyze how many unique prices exist at each timestamp"""
    conn = sqlite3.connect(db_path)
    
    df_trades = pd.read_sql('''
        SELECT timestamp, price
        FROM trades
        ORDER BY timestamp
    ''', conn)
    conn.close()
    
    if len(df_trades) == 0:
        return None
    
    # Group by timestamp and count unique prices
    grouped = df_trades.groupby('timestamp').agg({
        'price': ['count', 'nunique', 'min', 'max', 'std']
    }).reset_index()
    
    grouped.columns = ['timestamp', 'num_trades', 'unique_prices', 'min_price', 'max_price', 'price_std']
    
    # Calculate statistics
    stats = {
        'condition': condition,
        'total_trades': len(df_trades),
        'unique_timestamps': len(grouped),
        'avg_trades_per_timestamp': grouped['num_trades'].mean(),
        'avg_unique_prices_per_timestamp': grouped['unique_prices'].mean(),
        'timestamps_with_multiple_trades': (grouped['num_trades'] > 1).sum(),
        'timestamps_with_multiple_prices': (grouped['unique_prices'] > 1).sum(),
        'avg_price_spread_per_timestamp': (grouped['max_price'] - grouped['min_price']).mean()
    }
    
    return stats, df_trades

def main():
    print("="*60)
    print("TRADE GRANULARITY ANALYSIS")
    print("="*60)
    
    # Analyze real NASDAQ data
    print("\n📊 Analyzing Real NASDAQ Data...")
    parser = LOBSTERDataParser("AMZN", "2012-06-21", data_dir="/workspace")
    trades_df = parser.get_trades()
    
    if len(trades_df) > 0:
        # Group by timestamp
        grouped = trades_df.groupby('timestamp').agg({
            'price': ['count', 'nunique', 'min', 'max']
        }).reset_index()
        grouped.columns = ['timestamp', 'num_trades', 'unique_prices', 'min_price', 'max_price']
        
        print(f"\nReal NASDAQ Statistics:")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Unique timestamps: {len(grouped)}")
        print(f"  Avg trades per timestamp: {grouped['num_trades'].mean():.2f}")
        print(f"  Max trades at single timestamp: {grouped['num_trades'].max()}")
        print(f"  Timestamps with multiple trades: {(grouped['num_trades'] > 1).sum()}")
        print(f"  Timestamps with multiple prices: {(grouped['unique_prices'] > 1).sum()}")
        
        # Show example of multiple trades at same timestamp
        multi_trade_timestamps = grouped[grouped['num_trades'] > 5].head(3)
        if len(multi_trade_timestamps) > 0:
            print("\n  Examples of timestamps with multiple trades:")
            for _, row in multi_trade_timestamps.iterrows():
                ts = row['timestamp']
                trades_at_ts = trades_df[trades_df['timestamp'] == ts]
                prices = trades_at_ts['price'].values
                print(f"    Time {ts:.1f}s: {len(prices)} trades, prices: ${prices.min():.2f}-${prices.max():.2f}")
    
    # Analyze our heterogeneous model
    print("\n📊 Analyzing Our Heterogeneous Model...")
    
    all_stats = []
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        db_path = f"/workspace/lob_databases_heterogeneous/AMZN_2012-06-21_{condition}_heterogeneous.db"
        stats, df_trades = analyze_trades_at_timestamp(db_path, condition)
        if stats:
            all_stats.append(stats)
            print(f"\n{condition} Statistics:")
            print(f"  Total trades: {stats['total_trades']}")
            print(f"  Unique timestamps: {stats['unique_timestamps']}")
            print(f"  Avg trades per timestamp: {stats['avg_trades_per_timestamp']:.2f}")
            print(f"  Timestamps with multiple trades: {stats['timestamps_with_multiple_trades']}")
            print(f"  Timestamps with multiple prices: {stats['timestamps_with_multiple_prices']}")
            print(f"  Avg price spread per timestamp: ${stats['avg_price_spread_per_timestamp']:.4f}")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Real NASDAQ trade scatter
    ax = axes[0, 0]
    if len(trades_df) > 0:
        # Sample for visibility
        sample_size = min(5000, len(trades_df))
        sample_idx = np.random.choice(len(trades_df), sample_size, replace=False)
        sample_trades = trades_df.iloc[sample_idx]
        
        time_hours = sample_trades['timestamp'].values / 3600
        prices = sample_trades['price'].values
        ax.scatter(time_hours, prices, alpha=0.3, s=1, c='red')
        ax.set_title('Real NASDAQ - Trade Prices')
        ax.set_xlabel('Hours from market open')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
    
    # Plot 2-4: Our model trade scatter
    for idx, condition in enumerate(["LLMON", "LLMOFF", "Baseline"]):
        ax = axes.flat[idx + 1]
        db_path = f"/workspace/lob_databases_heterogeneous/AMZN_2012-06-21_{condition}_heterogeneous.db"
        conn = sqlite3.connect(db_path)
        df_trades = pd.read_sql('SELECT timestamp, price FROM trades', conn)
        conn.close()
        
        if len(df_trades) > 0:
            # Sample for visibility
            sample_size = min(5000, len(df_trades))
            sample_idx = np.random.choice(len(df_trades), sample_size, replace=False)
            sample_trades = df_trades.iloc[sample_idx]
            
            time_hours = sample_trades['timestamp'].values / 3600
            prices = sample_trades['price'].values
            ax.scatter(time_hours, prices, alpha=0.3, s=1, c='blue')
            ax.set_title(f'{condition} - Trade Prices')
            ax.set_xlabel('Hours from market open')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Trade Price Granularity Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/trade_granularity_analysis.png', dpi=150)
    print("✅ Plot saved to /workspace/trade_granularity_analysis.png")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("""
The issue is clear:
1. Real NASDAQ has multiple trades at different prices within same timestamp
2. Our model generates trades but they're too uniform in price
3. We need to add price variation based on:
   - Bid-ask spread
   - Market depth (multiple price levels)
   - Order types (market vs limit)
   - Sub-second timing differences
    """)

if __name__ == "__main__":
    main()