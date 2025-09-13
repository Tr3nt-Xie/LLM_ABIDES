#!/usr/bin/env python3
"""
LOB Comparison Experiment: LLMON vs LLMOFF vs Baseline vs Real NASDAQ
======================================================================

This module implements the three-condition experiment comparing:
1. LLMON: LLM-enhanced agents with news
2. LLMOFF: Same agents without LLM (rule-based)
3. Baseline: Traditional ABIDES agents
4. Real: Actual NASDAQ ITCH data

Produces ABIDES Figure 4 style visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from dataclasses import dataclass
# Set style for publication-quality figures
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
except ImportError:
    # Fallback if seaborn not available
    plt.style.use('default')
    pass

logger = logging.getLogger(__name__)

@dataclass
class LOBData:
    """Container for LOB data from each experimental condition"""
    condition: str  # LLMON, LLMOFF, Baseline, Real
    timestamps: np.ndarray
    mid_prices: np.ndarray
    trade_prices: np.ndarray
    trade_times: np.ndarray
    spreads: np.ndarray
    
    def get_time_series(self, start_time: float = None, end_time: float = None):
        """Get data within time window"""
        if start_time is None:
            start_time = self.timestamps[0] if len(self.timestamps) > 0 else 0
        if end_time is None:
            end_time = self.timestamps[-1] if len(self.timestamps) > 0 else 1e9
            
        # For simulated data (0-based timestamps)
        if self.timestamps[0] < 1000:  # Likely seconds from start, not absolute
            mask = (self.timestamps >= 0) & (self.timestamps <= self.timestamps[-1])
            trade_mask = (self.trade_times >= 0) & (self.trade_times <= self.trade_times[-1]) if len(self.trade_times) > 0 else np.array([], dtype=bool)
        else:
            # For real NASDAQ data (seconds after midnight)
            mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
            trade_mask = (self.trade_times >= start_time) & (self.trade_times <= end_time) if len(self.trade_times) > 0 else np.array([], dtype=bool)
        
        return {
            'timestamps': self.timestamps[mask],
            'mid_prices': self.mid_prices[mask],
            'trade_times': self.trade_times[trade_mask],
            'trade_prices': self.trade_prices[trade_mask],
            'spreads': self.spreads[mask]
        }

class LOBComparator:
    """Compares LOB data across experimental conditions"""
    
    def __init__(self, symbol: str, date: str, data_dir: str = "/workspace"):
        """
        Initialize comparator
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            date: Date in YYYY-MM-DD format
            data_dir: Directory containing data files
        """
        self.symbol = symbol
        self.date = date
        self.data_dir = Path(data_dir)
        self.lob_data = {}
        
    def load_real_nasdaq_data(self) -> LOBData:
        """Load real NASDAQ ITCH data"""
        
        # Import the ITCH parser
        import sys
        sys.path.insert(0, str(self.data_dir / "src"))
        from itch_data_parser import LOBSTERDataParser
        
        logger.info(f"Loading real NASDAQ data for {self.symbol} on {self.date}")
        
        parser = LOBSTERDataParser(self.symbol, self.date, self.data_dir)
        messages = parser.parse_messages()
        orderbook = parser.parse_orderbook()
        trades = parser.get_trades()
        
        # Extract time series
        timestamps = np.array([ob.timestamp for ob in orderbook])
        mid_prices = np.array([ob.mid_price for ob in orderbook if ob.mid_price > 0])
        spreads = np.array([ob.spread for ob in orderbook if ob.spread > 0])
        
        # Align timestamps for mid_prices and spreads
        valid_indices = [i for i, ob in enumerate(orderbook) if ob.mid_price > 0]
        timestamps_valid = timestamps[valid_indices]
        
        # Extract trade data
        if len(trades) > 0:
            trade_times = trades['timestamp'].values
            trade_prices = trades['price'].values
        else:
            trade_times = np.array([])
            trade_prices = np.array([])
        
        return LOBData(
            condition="Real NASDAQ",
            timestamps=timestamps_valid,
            mid_prices=mid_prices,
            trade_prices=trade_prices,
            trade_times=trade_times,
            spreads=spreads[:len(timestamps_valid)]  # Ensure same length
        )
    
    def load_simulated_data(self, condition: str, file_path: str) -> LOBData:
        """
        Load simulated LOB data from file
        
        Args:
            condition: Experimental condition name
            file_path: Path to simulation output file
        """
        
        logger.info(f"Loading simulated data for condition: {condition}")
        
        # This is a placeholder - adapt based on your actual output format
        # Assuming JSON format with timestamps, prices, trades
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract relevant fields (adjust based on actual format)
        timestamps = np.array(data.get('timestamps', []))
        mid_prices = np.array(data.get('mid_prices', []))
        trade_times = np.array(data.get('trade_times', []))
        trade_prices = np.array(data.get('trade_prices', []))
        spreads = np.array(data.get('spreads', []))
        
        return LOBData(
            condition=condition,
            timestamps=timestamps,
            mid_prices=mid_prices,
            trade_prices=trade_prices,
            trade_times=trade_times,
            spreads=spreads
        )
    
    def create_abides_figure4_plot(self, 
                                   start_time: float = None,
                                   end_time: float = None,
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: str = None):
        """
        Create ABIDES Figure 4 style visualization
        
        Creates a 2x2 subplot with:
        - Real NASDAQ data (top-left)
        - LLMON condition (top-right)
        - LLMOFF condition (bottom-left)
        - Baseline condition (bottom-right)
        
        Each subplot shows:
        - Mid-price as smooth blue line
        - Trade prices as red scatter points
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        fig.suptitle(f'{self.symbol} - LOB Comparison Experiment ({self.date})', 
                     fontsize=16, fontweight='bold')
        
        # Define plot positions for each condition
        plot_config = [
            (0, 0, 'Real NASDAQ'),
            (0, 1, 'LLMON'),
            (1, 0, 'LLMOFF'),
            (1, 1, 'Baseline')
        ]
        
        # Common y-axis limits (will be set after plotting all data)
        y_min, y_max = float('inf'), float('-inf')
        
        for row, col, condition in plot_config:
            ax = axes[row, col]
            
            if condition in self.lob_data:
                data = self.lob_data[condition]
                series = data.get_time_series(start_time, end_time)
                
                # Convert timestamps to hours for better readability
                if len(series['timestamps']) > 0:
                    time_hours = (series['timestamps'] - series['timestamps'][0]) / 3600
                else:
                    time_hours = np.array([])
                
                if len(series['trade_times']) > 0 and len(series['timestamps']) > 0:
                    trade_time_hours = (series['trade_times'] - series['timestamps'][0]) / 3600
                else:
                    trade_time_hours = np.array([])
                
                # Plot mid-price as smooth blue line
                ax.plot(time_hours, series['mid_prices'], 
                       color='blue', linewidth=1.5, alpha=0.8, label='Mid-price')
                
                # Plot trade prices as red scatter points
                if len(series['trade_prices']) > 0:
                    ax.scatter(trade_time_hours, series['trade_prices'],
                             color='red', s=10, alpha=0.6, label='Trades', zorder=5)
                
                # Update y-axis limits
                if len(series['mid_prices']) > 0:
                    y_min = min(y_min, np.min(series['mid_prices']) * 0.999)
                    y_max = max(y_max, np.max(series['mid_prices']) * 1.001)
                if len(series['trade_prices']) > 0:
                    y_min = min(y_min, np.min(series['trade_prices']) * 0.999)
                    y_max = max(y_max, np.max(series['trade_prices']) * 1.001)
                
                # Add statistics text
                if len(series['trade_prices']) > 0:
                    num_trades = len(series['trade_prices'])
                    avg_spread = np.mean(data.spreads) if len(data.spreads) > 0 else 0
                    
                    stats_text = f'Trades: {num_trades}\nAvg Spread: ${avg_spread:.4f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # No data for this condition
                ax.text(0.5, 0.5, 'No Data Available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
            
            # Set subplot title and labels
            ax.set_title(condition, fontweight='bold')
            ax.set_xlabel('Time (hours from market open)')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        # Set common y-axis limits for all subplots
        if y_min < float('inf'):
            for ax in axes.flat:
                ax.set_ylim(y_min, y_max)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig, axes
    
    def create_detailed_comparison_plot(self,
                                       start_time: float = None,
                                       end_time: float = None,
                                       figsize: Tuple[int, int] = (15, 12),
                                       save_path: str = None):
        """
        Create a more detailed comparison with additional metrics
        
        Includes:
        - Price series comparison
        - Spread evolution
        - Trade intensity
        - Price volatility
        """
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
        
        # Main price comparison (top row, spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        
        # Spread comparison (middle left)
        ax_spread = fig.add_subplot(gs[1, 0])
        
        # Trade intensity (middle right)
        ax_intensity = fig.add_subplot(gs[1, 1])
        
        # Return distribution (bottom left)
        ax_returns = fig.add_subplot(gs[2, 0])
        
        # Autocorrelation (bottom right)
        ax_acf = fig.add_subplot(gs[2, 1])
        
        colors = {'Real NASDAQ': 'black', 'LLMON': 'blue', 
                 'LLMOFF': 'orange', 'Baseline': 'green'}
        
        # Plot main price series
        for condition, color in colors.items():
            if condition in self.lob_data:
                data = self.lob_data[condition]
                series = data.get_time_series(start_time, end_time)
                
                time_hours = (series['timestamps'] - series['timestamps'][0]) / 3600
                
                # Plot mid-prices
                ax_main.plot(time_hours, series['mid_prices'],
                           color=color, linewidth=1.5, alpha=0.7, label=condition)
                
                # Calculate and plot spread
                if len(data.spreads) > 0:
                    spread_time = (data.timestamps[:len(data.spreads)] - data.timestamps[0]) / 3600
                    ax_spread.plot(spread_time, data.spreads,
                                 color=color, linewidth=1, alpha=0.7, label=condition)
                
                # Calculate trade intensity (trades per minute)
                if len(series['trade_times']) > 0:
                    bins = np.arange(0, time_hours[-1], 1/60)  # 1-minute bins
                    hist, bin_edges = np.histogram(
                        (series['trade_times'] - series['timestamps'][0]) / 3600, bins=bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax_intensity.plot(bin_centers, hist, 
                                    color=color, linewidth=1, alpha=0.7, label=condition)
                
                # Calculate returns
                if len(series['mid_prices']) > 1:
                    returns = np.diff(np.log(series['mid_prices']))
                    ax_returns.hist(returns, bins=30, alpha=0.5, color=color,
                                  label=condition, density=True)
                
                # Calculate autocorrelation of returns
                if len(series['mid_prices']) > 10:
                    returns = np.diff(np.log(series['mid_prices']))
                    lags = range(1, min(21, len(returns)//2))
                    acf_values = [np.corrcoef(returns[:-lag], returns[lag:])[0, 1] 
                                 for lag in lags]
                    ax_acf.plot(lags, acf_values, 'o-', 
                              color=color, markersize=4, linewidth=1,
                              alpha=0.7, label=condition)
        
        # Configure subplots
        ax_main.set_title(f'{self.symbol} - Price Series Comparison', fontweight='bold')
        ax_main.set_xlabel('Time (hours from market open)')
        ax_main.set_ylabel('Price ($)')
        ax_main.legend(loc='best')
        ax_main.grid(True, alpha=0.3)
        
        ax_spread.set_title('Bid-Ask Spread Evolution')
        ax_spread.set_xlabel('Time (hours)')
        ax_spread.set_ylabel('Spread ($)')
        ax_spread.legend(loc='best', fontsize=9)
        ax_spread.grid(True, alpha=0.3)
        
        ax_intensity.set_title('Trade Intensity (trades/minute)')
        ax_intensity.set_xlabel('Time (hours)')
        ax_intensity.set_ylabel('Trades per minute')
        ax_intensity.legend(loc='best', fontsize=9)
        ax_intensity.grid(True, alpha=0.3)
        
        ax_returns.set_title('Return Distribution')
        ax_returns.set_xlabel('Log Returns')
        ax_returns.set_ylabel('Density')
        ax_returns.legend(loc='best', fontsize=9)
        ax_returns.grid(True, alpha=0.3)
        
        ax_acf.set_title('Return Autocorrelation')
        ax_acf.set_xlabel('Lag')
        ax_acf.set_ylabel('Autocorrelation')
        ax_acf.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_acf.legend(loc='best', fontsize=9)
        ax_acf.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.symbol} - Comprehensive LOB Analysis ({self.date})',
                    fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detailed figure saved to {save_path}")
        
        return fig
    
    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate comparison metrics for all conditions"""
        
        metrics = []
        
        for condition, data in self.lob_data.items():
            if len(data.mid_prices) > 1:
                returns = np.diff(np.log(data.mid_prices))
                
                metric_dict = {
                    'Condition': condition,
                    'Num_Trades': len(data.trade_prices),
                    'Avg_Spread': np.mean(data.spreads) if len(data.spreads) > 0 else np.nan,
                    'Std_Spread': np.std(data.spreads) if len(data.spreads) > 0 else np.nan,
                    'Price_Mean': np.mean(data.mid_prices),
                    'Price_Std': np.std(data.mid_prices),
                    'Return_Mean': np.mean(returns) * 10000,  # in basis points
                    'Return_Std': np.std(returns) * 10000,  # in basis points
                    'Return_Skew': pd.Series(returns).skew(),
                    'Return_Kurt': pd.Series(returns).kurt(),
                    'Price_Range': np.max(data.mid_prices) - np.min(data.mid_prices)
                }
                
                metrics.append(metric_dict)
        
        return pd.DataFrame(metrics)

def generate_mock_simulated_data(symbol: str, date: str, condition: str, 
                                base_price: float = 100.0, 
                                num_points: int = 1000) -> Dict:
    """
    Generate mock simulated data for testing
    (Replace this with actual simulation output)
    """
    
    # Generate timestamps (seconds from market open)
    timestamps = np.linspace(0, 6.5 * 3600, num_points)  # 6.5 hour trading day
    
    # Generate mid-prices with random walk
    returns = np.random.normal(0, 0.0001, num_points)
    
    # Add condition-specific characteristics
    if condition == "LLMON":
        # LLM-enhanced: smoother, more reactive to events
        returns = np.convolve(returns, np.ones(5)/5, mode='same')
        # Add news event impact
        event_time = num_points // 2
        returns[event_time:event_time+50] += np.linspace(0, 0.001, min(50, num_points-event_time))
    elif condition == "LLMOFF":
        # Without LLM: more noise, less smooth
        returns *= 1.2
    else:  # Baseline
        # Traditional: moderate noise
        returns *= 1.1
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate trades (subset of timestamps)
    num_trades = num_points // 10
    trade_indices = np.sort(np.random.choice(num_points, num_trades, replace=False))
    trade_times = timestamps[trade_indices]
    trade_prices = prices[trade_indices] + np.random.normal(0, 0.01, num_trades)
    
    # Generate spreads
    spreads = np.random.gamma(2, 0.01, num_points) + 0.01
    
    return {
        'timestamps': timestamps.tolist(),
        'mid_prices': prices.tolist(),
        'trade_times': trade_times.tolist(),
        'trade_prices': trade_prices.tolist(),
        'spreads': spreads.tolist()
    }

def main():
    """Example usage of the LOB comparison experiment"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize comparator
    symbol = "AMZN"
    date = "2012-06-21"
    comparator = LOBComparator(symbol, date)
    
    # Load real NASDAQ data
    try:
        real_data = comparator.load_real_nasdaq_data()
        comparator.lob_data['Real NASDAQ'] = real_data
        logger.info(f"Loaded real NASDAQ data: {len(real_data.mid_prices)} points")
    except Exception as e:
        logger.error(f"Failed to load real NASDAQ data: {e}")
        # Generate mock data for demonstration
        mock_real = generate_mock_simulated_data(symbol, date, "Real", base_price=223.0)
        comparator.lob_data['Real NASDAQ'] = LOBData(
            condition="Real NASDAQ",
            timestamps=np.array(mock_real['timestamps']),
            mid_prices=np.array(mock_real['mid_prices']),
            trade_prices=np.array(mock_real['trade_prices']),
            trade_times=np.array(mock_real['trade_times']),
            spreads=np.array(mock_real['spreads'])
        )
    
    # Generate mock simulated data for other conditions
    # (Replace with actual simulation outputs)
    for condition in ['LLMON', 'LLMOFF', 'Baseline']:
        mock_data = generate_mock_simulated_data(
            symbol, date, condition, 
            base_price=223.0 if symbol == "AMZN" else 100.0
        )
        comparator.lob_data[condition] = LOBData(
            condition=condition,
            timestamps=np.array(mock_data['timestamps']),
            mid_prices=np.array(mock_data['mid_prices']),
            trade_prices=np.array(mock_data['trade_prices']),
            trade_times=np.array(mock_data['trade_times']),
            spreads=np.array(mock_data['spreads'])
        )
    
    # Create ABIDES Figure 4 style plot
    output_dir = Path("/workspace/artifacts/lob_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = comparator.create_abides_figure4_plot(
        save_path=output_dir / f"{symbol}_{date}_abides_figure4.png"
    )
    
    # Create detailed comparison plot
    fig_detailed = comparator.create_detailed_comparison_plot(
        save_path=output_dir / f"{symbol}_{date}_detailed_comparison.png"
    )
    
    # Calculate and display metrics
    metrics_df = comparator.calculate_metrics()
    print("\n📊 Comparison Metrics:")
    print(metrics_df.to_string())
    
    # Save metrics
    metrics_df.to_csv(output_dir / f"{symbol}_{date}_metrics.csv", index=False)
    logger.info(f"Metrics saved to {output_dir / f'{symbol}_{date}_metrics.csv'}")
    
    plt.show()

if __name__ == "__main__":
    main()