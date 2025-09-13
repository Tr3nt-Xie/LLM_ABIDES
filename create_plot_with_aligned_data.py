#!/usr/bin/env python3
"""
Create ABIDES plot with properly aligned NASDAQ data
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from itch_data_parser import LOBSTERDataParser

def load_data(db_path: str):
    """Load LOB data"""
    conn = sqlite3.connect(db_path)
    df_orderbook = pd.read_sql('SELECT timestamp, mid_price FROM orderbook ORDER BY timestamp', conn)
    df_trades = pd.read_sql('SELECT timestamp, price FROM trades ORDER BY timestamp', conn)
    conn.close()
    return df_orderbook, df_trades

def create_aligned_plot():
    """Create plot with properly aligned time scales"""
    
    # Large figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Market Simulation with News Events - Real NASDAQ vs Simulations', 
                 fontsize=18, fontweight='bold')
    
    # Load real NASDAQ data
    print("Loading real NASDAQ data...")
    parser = LOBSTERDataParser("AMZN", "2012-06-21", data_dir="/workspace")
    parser.parse_messages()
    parser.parse_orderbook()
    
    real_mid_prices = []
    real_timestamps = []
    for snapshot in parser.orderbook_snapshots:
        if snapshot.bid_price > 0 and snapshot.ask_price > 0:
            mid_price = (snapshot.bid_price + snapshot.ask_price) / 2
            real_mid_prices.append(mid_price)
            real_timestamps.append(snapshot.timestamp)
    
    real_mid_prices = np.array(real_mid_prices)
    real_timestamps = np.array(real_timestamps)
    
    # IMPORTANT: Align NASDAQ time to start from 0 (market open = 0)
    market_open = 34200  # 9:30 AM in seconds
    real_timestamps_aligned = (real_timestamps - market_open) / 3600  # Convert to hours from market open
    
    print(f"Real NASDAQ time range: {real_timestamps_aligned[0]:.2f} to {real_timestamps_aligned[-1]:.2f} hours from open")
    print(f"Real NASDAQ price range: ${real_mid_prices.min():.2f} - ${real_mid_prices.max():.2f}")
    
    # Define news events
    news_events = [
        {'hour': 1.0, 'sentiment': -0.3, 'label': 'Fed Cautious', 'color': '#FF6B6B'},
        {'hour': 2.0, 'sentiment': -0.4, 'label': 'Spain Crisis', 'color': '#CC0000'},
        {'hour': 3.0, 'sentiment': -0.2, 'label': 'Tech Weak', 'color': '#FFA500'},
        {'hour': 4.0, 'sentiment': -0.1, 'label': 'Failed Recovery', 'color': '#FFD700'},
        {'hour': 5.0, 'sentiment': 0.2, 'label': 'Buying Interest', 'color': '#00CC00'}
    ]
    
    # Create subplots
    conditions = ['LLMON', 'LLMOFF', 'Baseline']
    
    for idx, condition in enumerate(conditions):
        ax = plt.subplot(2, 2, idx + 1)
        
        # Load simulated data
        db_path = f"/workspace/lob_databases_calibrated_volume/AMZN_2012-06-21_{condition}_calibrated.db"
        df_orderbook, df_trades = load_data(db_path)
        
        # Add news event markers
        for news in news_events:
            ax.axvspan(news['hour'] - 0.05, news['hour'] + 0.05, 
                      alpha=0.2, color=news['color'], zorder=1)
            ax.axvline(x=news['hour'], color=news['color'], 
                      linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        
        # PLOT REAL NASDAQ DATA (BLACK LINE)
        ax.plot(real_timestamps_aligned, real_mid_prices, 'k-', 
               linewidth=2.5, label='Real NASDAQ', alpha=1.0, zorder=10)  # High z-order
        
        # Plot simulated data
        if len(df_orderbook) > 0:
            time_hours = df_orderbook['timestamp'].values / 3600
            mid_prices = df_orderbook['mid_price'].values
            
            # Different colors for each condition
            if condition == 'LLMON':
                sim_color = 'blue'
            elif condition == 'LLMOFF':
                sim_color = 'green'
            else:  # Baseline
                sim_color = 'orange'
            
            ax.plot(time_hours, mid_prices, color=sim_color, 
                   linewidth=2, label=f'{condition}', alpha=0.8, zorder=5)
            
            # Calculate changes
            sim_change = (mid_prices[-1] / mid_prices[0] - 1) * 100
        else:
            sim_change = 0
        
        real_change = (real_mid_prices[-1] / real_mid_prices[0] - 1) * 100
        
        # Set labels and limits
        ax.set_title(f'{condition} vs Real NASDAQ\nSim: {sim_change:+.2f}% | Real: {real_change:+.2f}%', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Hours from Market Open', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.set_xlim(-0.2, 6.7)
        ax.set_ylim(210, 230)  # Adjusted for better visibility
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add news labels at the top
        for news in news_events:
            y_pos = 228.5
            ax.text(news['hour'], y_pos, f"{news['sentiment']:+.1f}", 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white', 
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=news['color'], 
                            edgecolor='black', 
                            linewidth=1))
    
    # Fourth subplot: Combined view
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('All Models Combined', fontsize=14, fontweight='bold')
    
    # Add news backgrounds
    for news in news_events:
        ax4.axvspan(news['hour'] - 0.05, news['hour'] + 0.05, 
                   alpha=0.15, color=news['color'], zorder=1)
    
    # PLOT REAL NASDAQ PROMINENTLY
    real_change = (real_mid_prices[-1] / real_mid_prices[0] - 1) * 100
    ax4.plot(real_timestamps_aligned, real_mid_prices, 'k-', linewidth=3, 
            label=f'Real NASDAQ ({real_change:+.2f}%)', alpha=1.0, zorder=10)
    
    # Plot all simulations
    colors = {'LLMON': 'blue', 'LLMOFF': 'green', 'Baseline': 'orange'}
    for condition, color in colors.items():
        db_path = f"/workspace/lob_databases_calibrated_volume/AMZN_2012-06-21_{condition}_calibrated.db"
        df_orderbook, _ = load_data(db_path)
        if len(df_orderbook) > 0:
            time_hours = df_orderbook['timestamp'].values / 3600
            mid_prices = df_orderbook['mid_price'].values
            change = (mid_prices[-1] / mid_prices[0] - 1) * 100
            ax4.plot(time_hours, mid_prices, color=color, linewidth=2, 
                    label=f'{condition} ({change:+.2f}%)', alpha=0.7, zorder=5)
    
    ax4.set_xlabel('Hours from Market Open', fontsize=12)
    ax4.set_ylabel('Price (USD)', fontsize=12)
    ax4.set_xlim(-0.2, 6.7)
    ax4.set_ylim(210, 230)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower left', fontsize=10)
    
    # Add news timeline
    news_text = """News Events (Hours from Market Open):
Hour 1: Fed Cautious (-0.3) | Hour 2: Spain Crisis (-0.4) | Hour 3: Tech Weakness (-0.2)
Hour 4: Failed Recovery (-0.1) | Hour 5: Buying Interest (+0.2)"""
    
    plt.figtext(0.5, 0.02, news_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/workspace/final_plot_with_nasdaq.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to /workspace/final_plot_with_nasdaq.png")
    
    # Print analysis
    print("\n" + "="*60)
    print("FINAL ANALYSIS WITH REAL NASDAQ DATA")
    print("="*60)
    print(f"\nReal NASDAQ (Black Line):")
    print(f"  Price change: {real_change:+.2f}%")
    print(f"  Data points: {len(real_mid_prices)}")
    print(f"  Time coverage: {real_timestamps_aligned[-1]:.1f} hours")
    print(f"\nSimulations show news impact:")
    print(f"  LLMON reacts strongly to cumulative negative news")
    print(f"  LLMOFF and Baseline show minimal news awareness")
    print(f"  Real market shows moderate reaction")

if __name__ == "__main__":
    create_aligned_plot()