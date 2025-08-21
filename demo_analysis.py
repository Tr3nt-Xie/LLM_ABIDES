#!/usr/bin/env python3
"""
Market Simulator Analysis Demo
=============================

This script demonstrates how to analyze the generated market simulation data
from the ABIDES-LLM integration project.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def analyze_database(db_path):
    """Analyze the generated market simulation database"""
    print("🔍 Analyzing Market Simulation Database")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get table info
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"📊 Tables in database: {', '.join(tables['name'].tolist())}")
    
    # Analyze orders
    print("\n📈 ORDER ANALYSIS")
    print("-" * 30)
    
    if 'detailed_orders' in tables['name'].values:
        orders_df = pd.read_sql_query("""
            SELECT agent_type, symbol, side, COUNT(*) as count, 
                   AVG(price) as avg_price, AVG(quantity) as avg_quantity
            FROM detailed_orders 
            GROUP BY agent_type, symbol, side
            ORDER BY count DESC
        """, conn)
        
        print("Top order types:")
        print(orders_df.head(10))
        
        # Order distribution by agent type
        agent_dist = pd.read_sql_query("""
            SELECT agent_type, COUNT(*) as order_count
            FROM detailed_orders
            GROUP BY agent_type
            ORDER BY order_count DESC
        """, conn)
        
        print(f"\n📊 Agent Order Distribution:")
        for _, row in agent_dist.iterrows():
            print(f"  {row['agent_type']}: {row['order_count']:,} orders")
    
    # Analyze trades
    print("\n💹 TRADE ANALYSIS")
    print("-" * 30)
    
    if 'detailed_trades' in tables['name'].values:
        trades_df = pd.read_sql_query("""
            SELECT symbol, COUNT(*) as trade_count, 
                   AVG(price) as avg_price, AVG(quantity) as avg_quantity,
                   MIN(price) as min_price, MAX(price) as max_price
            FROM detailed_trades
            GROUP BY symbol
        """, conn)
        
        print("Trade summary by symbol:")
        print(trades_df)
        
        # Price movement analysis
        price_moves = pd.read_sql_query("""
            SELECT symbol, timestamp, price
            FROM detailed_trades
            ORDER BY symbol, timestamp
            LIMIT 1000
        """, conn)
        
        if not price_moves.empty:
            print(f"\n📈 Price Range Analysis:")
            for symbol in price_moves['symbol'].unique():
                symbol_data = price_moves[price_moves['symbol'] == symbol]
                price_range = symbol_data['price'].max() - symbol_data['price'].min()
                print(f"  {symbol}: ${symbol_data['price'].min():.2f} - ${symbol_data['price'].max():.2f} (range: ${price_range:.2f})")
    
    # Market microstructure analysis
    if 'lob_snapshots' in tables['name'].values:
        print("\n📊 MARKET MICROSTRUCTURE")
        print("-" * 30)
        
        snapshots_count = pd.read_sql_query("SELECT COUNT(*) as count FROM lob_snapshots", conn)
        print(f"Total snapshots: {snapshots_count['count'].iloc[0]:,}")
        
        # Check available columns first
        try:
            columns_query = pd.read_sql_query("PRAGMA table_info(lob_snapshots)", conn)
            available_columns = columns_query['name'].tolist()
            print(f"Available columns: {', '.join(available_columns)}")
            
            # Try to get basic snapshot info
            sample_data = pd.read_sql_query("""
                SELECT symbol, COUNT(*) as snapshot_count
                FROM lob_snapshots
                GROUP BY symbol
            """, conn)
            
            if not sample_data.empty:
                print("\nSnapshot distribution:")
                print(sample_data)
        except Exception as e:
            print(f"Error analyzing snapshots: {e}")
    
    conn.close()
    print(f"\n✅ Database analysis completed!")

def create_sample_visualization(db_path):
    """Create sample visualizations of the market data"""
    print("\n📊 Creating Sample Visualizations")
    print("=" * 40)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Get trade data for visualization
        trades_df = pd.read_sql_query("""
            SELECT timestamp, symbol, price, quantity
            FROM detailed_trades
            ORDER BY timestamp
            LIMIT 5000
        """, conn)
        
        if trades_df.empty:
            print("❌ No trade data available for visualization")
            return
        
        # Convert timestamp to datetime
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Create price chart
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Price over time
        plt.subplot(2, 2, 1)
        for symbol in trades_df['symbol'].unique():
            symbol_data = trades_df[trades_df['symbol'] == symbol]
            plt.plot(symbol_data['timestamp'], symbol_data['price'], 
                    label=f'{symbol}', alpha=0.7, linewidth=1)
        plt.title('Price Movement Over Time')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Trade volume
        plt.subplot(2, 2, 2)
        volume_data = trades_df.groupby('timestamp')['quantity'].sum().reset_index()
        plt.plot(volume_data['timestamp'], volume_data['quantity'], 'g-', alpha=0.7)
        plt.title('Trading Volume Over Time')
        plt.xlabel('Time')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Price distribution
        plt.subplot(2, 2, 3)
        for symbol in trades_df['symbol'].unique():
            symbol_data = trades_df[trades_df['symbol'] == symbol]
            plt.hist(symbol_data['price'], alpha=0.6, bins=30, label=f'{symbol}')
        plt.title('Price Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Trade size distribution
        plt.subplot(2, 2, 4)
        plt.hist(trades_df['quantity'], bins=50, alpha=0.7, color='orange')
        plt.title('Trade Size Distribution')
        plt.xlabel('Quantity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'market_analysis_demo.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
    finally:
        conn.close()

def main():
    """Main demonstration function"""
    print("🚀 ABIDES-LLM Market Simulator Analysis Demo")
    print("=" * 60)
    
    # Check for available databases
    db_files = [
        'test_lob.db',
        'quick_test_orderbook.db',
        'small_scale_lob.db'
    ]
    
    available_dbs = [db for db in db_files if os.path.exists(db)]
    
    if not available_dbs:
        print("❌ No simulation databases found!")
        print("Please run one of the following first:")
        print("  python enhanced_orderbook_main.py --config quick_test")
        print("  python scaled_lob_main.py --scale small")
        return
    
    print(f"📊 Found {len(available_dbs)} database(s): {', '.join(available_dbs)}")
    
    # Analyze the first available database
    db_path = available_dbs[0]
    print(f"\n🔍 Analyzing: {db_path}")
    
    analyze_database(db_path)
    create_sample_visualization(db_path)
    
    print("\n🎉 Demo completed!")
    print("\n📋 Next Steps:")
    print("1. Explore the generated databases with SQL queries")
    print("2. Run validation plots: python src/validation_viz.py --db <db_name> --symbol <SYMBOL>")
    print("3. Scale up simulations for larger datasets")
    print("4. Integrate with your own trading strategies")

if __name__ == "__main__":
    main()