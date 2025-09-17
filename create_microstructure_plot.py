#!/usr/bin/env python3
"""
Create ABIDES Figure 4 style plot with microstructure-aware trade dots
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from itch_data_parser import LOBSTERDataParser

def load_microstructure_data(db_path: str):
    """Load microstructure LOB data"""
    conn = sqlite3.connect(db_path)
    
    # Load orderbook for mid prices
    df_orderbook = pd.read_sql('''
        SELECT timestamp, mid_price
        FROM orderbook
        ORDER BY timestamp
    ''', conn)
    
    # Load trades
    df_trades = pd.read_sql('''
        SELECT timestamp, price
        FROM trades
        ORDER BY timestamp
    ''', conn)
    
    conn.close()
    
    return df_orderbook, df_trades

def create_abides_figure4_plot():
    """Create Figure 4 style plot with proper trade dots"""
    
    # Set up the plot with ABIDES paper style
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Market Microstructure Comparison - ABIDES Figure 4 Style', 
                 fontsize=14, fontweight='bold')
    
    # Load real NASDAQ data
    print("Loading real NASDAQ data...")
    parser = LOBSTERDataParser("AMZN", "2012-06-21", data_dir="/workspace")
    parser.parse_messages()
    parser.parse_orderbook()
    trades_df = parser.get_trades()
    
    # Calculate real mid prices
    real_mid_prices = []
    real_timestamps = []
    for snapshot in parser.orderbook_snapshots:
        if snapshot.bid_price > 0 and snapshot.ask_price > 0:
            mid_price = (snapshot.bid_price + snapshot.ask_price) / 2
            real_mid_prices.append(mid_price)
            real_timestamps.append(snapshot.timestamp)
    
    # Plot 1: Real NASDAQ
    ax = axes[0, 0]
    ax.set_title('Real NASDAQ ITCH Data', fontsize=12, fontweight='bold')
    
    # Plot mid-price as smooth line
    if len(real_timestamps) > 0:
        time_hours = np.array(real_timestamps) / 3600
        ax.plot(time_hours, real_mid_prices, 'b-', linewidth=1.5, label='Mid Price', alpha=0.8)
    
    # Plot trades as scatter points
    if len(trades_df) > 0:
        # Sample trades for visibility (too many points otherwise)
        sample_size = min(2000, len(trades_df))
        sample_idx = np.random.choice(len(trades_df), sample_size, replace=False)
        sample_trades = trades_df.iloc[sample_idx]
        
        trade_times = sample_trades['timestamp'].values / 3600
        trade_prices = sample_trades['price'].values
        ax.scatter(trade_times, trade_prices, c='red', s=2, alpha=0.5, label='Trades')
    
    ax.set_xlabel('Hours from Market Open')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Plot 2-4: Simulated conditions
    conditions = ["LLMON", "LLMOFF", "Baseline"]
    titles = ["LLMON (LLM Enhanced)", "LLMOFF (Traditional)", "Baseline"]
    
    for idx, (condition, title) in enumerate(zip(conditions, titles)):
        ax = axes.flat[idx + 1]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Load microstructure data
        db_path = f"/workspace/lob_databases_microstructure/AMZN_2012-06-21_{condition}_microstructure.db"
        df_orderbook, df_trades = load_microstructure_data(db_path)
        
        # Plot mid-price line
        if len(df_orderbook) > 0:
            time_hours = df_orderbook['timestamp'].values / 3600
            mid_prices = df_orderbook['mid_price'].values
            ax.plot(time_hours, mid_prices, 'b-', linewidth=1.5, label='Mid Price', alpha=0.8)
        
        # Plot trade scatter
        if len(df_trades) > 0:
            # Sample trades for visibility
            sample_size = min(2000, len(df_trades))
            sample_idx = np.random.choice(len(df_trades), sample_size, replace=False)
            sample_trades = df_trades.iloc[sample_idx]
            
            trade_times = sample_trades['timestamp'].values / 3600
            trade_prices = sample_trades['price'].values
            ax.scatter(trade_times, trade_prices, c='red', s=2, alpha=0.5, label='Trades')
        
        ax.set_xlabel('Hours from Market Open')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/workspace/abides_figure4_microstructure.png', dpi=150)
    print("✅ Figure saved to /workspace/abides_figure4_microstructure.png")
    
    # Create a zoomed-in view to show microstructure detail
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Market Microstructure Detail - 1 Minute Window', fontsize=14, fontweight='bold')
    
    # Zoom window: Hour 2.0 to 2.017 (1 minute)
    zoom_start = 2.0 * 3600
    zoom_end = 2.017 * 3600
    
    # Left: Real NASDAQ zoomed
    ax = axes2[0]
    ax.set_title('Real NASDAQ - 1 Minute Detail')
    
    if len(trades_df) > 0:
        zoom_trades = trades_df[(trades_df['timestamp'] >= zoom_start) & 
                                (trades_df['timestamp'] <= zoom_end)]
        if len(zoom_trades) > 0:
            trade_times = (zoom_trades['timestamp'].values - zoom_start) 
            trade_prices = zoom_trades['price'].values
            ax.scatter(trade_times, trade_prices, c='red', s=20, alpha=0.6)
            ax.set_xlim(0, 60)
            ax.set_xlabel('Seconds')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            
            # Show price levels
            unique_prices = np.unique(trade_prices)
            for price in unique_prices:
                ax.axhline(y=price, color='gray', linestyle=':', alpha=0.3)
    
    # Right: LLMON zoomed
    ax = axes2[1]
    ax.set_title('LLMON Simulation - 1 Minute Detail')
    
    db_path = f"/workspace/lob_databases_microstructure/AMZN_2012-06-21_LLMON_microstructure.db"
    conn = sqlite3.connect(db_path)
    zoom_trades = pd.read_sql(f'''
        SELECT timestamp, price 
        FROM trades 
        WHERE timestamp >= {zoom_start} AND timestamp <= {zoom_end}
        ORDER BY timestamp
    ''', conn)
    conn.close()
    
    if len(zoom_trades) > 0:
        trade_times = (zoom_trades['timestamp'].values - zoom_start)
        trade_prices = zoom_trades['price'].values
        ax.scatter(trade_times, trade_prices, c='red', s=20, alpha=0.6)
        ax.set_xlim(0, 60)
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        
        # Show price levels
        unique_prices = np.unique(trade_prices)
        for price in unique_prices[:20]:  # Limit to 20 levels for clarity
            ax.axhline(y=price, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/microstructure_detail.png', dpi=150)
    print("✅ Detail view saved to /workspace/microstructure_detail.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("MICROSTRUCTURE STATISTICS")
    print("="*60)
    
    for condition in ["LLMON", "LLMOFF", "Baseline"]:
        db_path = f"/workspace/lob_databases_microstructure/AMZN_2012-06-21_{condition}_microstructure.db"
        conn = sqlite3.connect(db_path)
        
        # Get trade statistics
        stats = pd.read_sql('''
            SELECT 
                COUNT(*) as total_trades,
                COUNT(DISTINCT CAST(timestamp AS INTEGER)) as unique_seconds,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM trades
        ''', conn)
        
        # Get trades per second distribution
        trades_per_sec = pd.read_sql('''
            SELECT 
                CAST(timestamp AS INTEGER) as second,
                COUNT(*) as num_trades,
                COUNT(DISTINCT price) as unique_prices,
                MAX(price) - MIN(price) as price_range
            FROM trades
            GROUP BY CAST(timestamp AS INTEGER)
        ''', conn)
        
        conn.close()
        
        print(f"\n{condition}:")
        print(f"  Total trades: {stats['total_trades'].iloc[0]:,}")
        print(f"  Avg trades/second: {trades_per_sec['num_trades'].mean():.1f}")
        print(f"  Max trades/second: {trades_per_sec['num_trades'].max()}")
        print(f"  Avg unique prices/second: {trades_per_sec['unique_prices'].mean():.1f}")
        print(f"  Avg price range/second: ${trades_per_sec['price_range'].mean():.3f}")

if __name__ == "__main__":
    create_abides_figure4_plot()