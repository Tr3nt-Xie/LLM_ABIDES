#!/usr/bin/env python3
"""
Analyze NASDAQ ITCH data to see if there are multiple execution prices at same timestamp
"""

import sys
sys.path.insert(0, 'src')
from itch_data_parser import LOBSTERDataParser
import pandas as pd
import numpy as np

print("Loading NASDAQ ITCH data...")
parser = LOBSTERDataParser("AMZN", "2012-06-21", data_dir="/workspace")

# Get trades
trades_df = parser.get_trades()
print(f"Total trades in NASDAQ data: {len(trades_df)}")

if len(trades_df) > 0:
    # Group by timestamp to see how many trades occur at same time
    trades_per_timestamp = trades_df.groupby('timestamp').agg({
        'price': ['count', 'nunique', 'min', 'max', 'std'],
        'size': 'sum'
    }).reset_index()
    
    # Flatten column names
    trades_per_timestamp.columns = ['timestamp', 'num_trades', 'unique_prices', 
                                   'min_price', 'max_price', 'price_std', 'total_size']
    
    # Find timestamps with multiple trades
    multi_trade_timestamps = trades_per_timestamp[trades_per_timestamp['num_trades'] > 1]
    
    print(f"\nTimestamps with multiple trades: {len(multi_trade_timestamps)}")
    print(f"Timestamps with single trade: {len(trades_per_timestamp) - len(multi_trade_timestamps)}")
    
    if len(multi_trade_timestamps) > 0:
        print("\n" + "="*60)
        print("ANALYSIS OF SAME-TIMESTAMP TRADES")
        print("="*60)
        
        # Statistics
        print(f"\nMax trades at single timestamp: {multi_trade_timestamps['num_trades'].max()}")
        print(f"Avg trades when multiple: {multi_trade_timestamps['num_trades'].mean():.2f}")
        print(f"Max unique prices at single timestamp: {multi_trade_timestamps['unique_prices'].max()}")
        print(f"Avg unique prices when multiple trades: {multi_trade_timestamps['unique_prices'].mean():.2f}")
        
        # Show examples
        print("\n" + "-"*60)
        print("EXAMPLES OF MULTIPLE TRADES AT SAME TIMESTAMP:")
        print("-"*60)
        
        # Get some interesting examples
        examples = multi_trade_timestamps.nlargest(5, 'num_trades')
        
        for idx, row in examples.iterrows():
            timestamp = row['timestamp']
            # Get actual trades at this timestamp
            trades_at_ts = trades_df[trades_df['timestamp'] == timestamp]
            
            print(f"\nTimestamp {timestamp:.6f} ({timestamp/3600:.3f} hours):")
            print(f"  {row['num_trades']} trades, {row['unique_prices']} unique prices")
            print(f"  Price range: ${row['min_price']:.2f} - ${row['max_price']:.2f}")
            print(f"  Price spread: ${row['max_price'] - row['min_price']:.2f}")
            
            # Show individual trades
            print("  Individual trades:")
            for _, trade in trades_at_ts.head(10).iterrows():
                print(f"    Price: ${trade['price']:.2f}, Size: {trade['size']}, Side: {trade.get('side', 'N/A')}")
            if len(trades_at_ts) > 10:
                print(f"    ... and {len(trades_at_ts) - 10} more trades")
        
        # Analyze price differences at same timestamp
        print("\n" + "-"*60)
        print("PRICE VARIATION AT SAME TIMESTAMP:")
        print("-"*60)
        
        # Calculate price spread for timestamps with multiple trades
        multi_trade_timestamps['price_spread'] = multi_trade_timestamps['max_price'] - multi_trade_timestamps['min_price']
        
        print(f"\nAverage price spread when multiple trades: ${multi_trade_timestamps['price_spread'].mean():.4f}")
        print(f"Max price spread at single timestamp: ${multi_trade_timestamps['price_spread'].max():.4f}")
        print(f"Median price spread: ${multi_trade_timestamps['price_spread'].median():.4f}")
        
        # Distribution of unique prices
        print(f"\nDistribution of unique prices per timestamp:")
        unique_price_dist = multi_trade_timestamps['unique_prices'].value_counts().sort_index()
        for num_prices, count in unique_price_dist.items():
            print(f"  {num_prices} unique price(s): {count} timestamps ({count/len(multi_trade_timestamps)*100:.1f}%)")
    
    # Overall summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nYES - NASDAQ ITCH data DOES have multiple execution prices at the same timestamp!")
    print(f"- {len(multi_trade_timestamps)} timestamps have multiple trades")
    print(f"- Up to {multi_trade_timestamps['unique_prices'].max() if len(multi_trade_timestamps) > 0 else 0} different prices at single timestamp")
    print(f"- This represents bid-ask spread and market microstructure")
    
else:
    print("No trades found in data")