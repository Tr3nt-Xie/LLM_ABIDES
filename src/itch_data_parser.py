#!/usr/bin/env python3
"""
NASDAQ ITCH/LOBSTER Data Parser and Integration
==============================================

This module provides functionality to parse NASDAQ ITCH data (via LOBSTER format)
and integrate it with the ABIDES-LLM simulator for realistic market replay and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """LOBSTER message types"""
    SUBMISSION = 1  # New limit order
    CANCELLATION = 2  # Partial deletion
    DELETION = 3  # Total deletion
    VISIBLE_EXECUTION = 4  # Execution of visible limit order
    HIDDEN_EXECUTION = 5  # Execution of hidden limit order
    TRADING_HALT = 7  # Trading halt indicator

@dataclass
class ITCHMessage:
    """Represents a single ITCH message"""
    timestamp: float  # Seconds after midnight
    msg_type: MessageType
    order_id: int
    size: int
    price: float  # Actual dollar price (converted from ITCH format)
    direction: int  # -1 for sell, 1 for buy
    
    @property
    def side(self) -> str:
        return "BUY" if self.direction == 1 else "SELL"
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime (assuming today's date)"""
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return base_date + timedelta(seconds=self.timestamp)

@dataclass
class OrderBookSnapshot:
    """Represents order book state at a point in time"""
    timestamp: float
    ask_price: float
    ask_size: int
    bid_price: float
    bid_size: int
    spread: float
    mid_price: float
    
    @classmethod
    def from_row(cls, row: pd.Series, timestamp: float) -> 'OrderBookSnapshot':
        """Create snapshot from LOBSTER orderbook row"""
        ask_price = row[0] / 10000  # Convert from ITCH price format
        ask_size = row[1]
        bid_price = row[2] / 10000
        bid_size = row[3]
        
        return cls(
            timestamp=timestamp,
            ask_price=ask_price,
            ask_size=ask_size,
            bid_price=bid_price,
            bid_size=bid_size,
            spread=ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0,
            mid_price=(ask_price + bid_price) / 2 if ask_price > 0 and bid_price > 0 else 0
        )

class LOBSTERDataParser:
    """Parser for LOBSTER format NASDAQ ITCH data"""
    
    def __init__(self, symbol: str, date: str, data_dir: str = "."):
        """
        Initialize parser for a specific symbol and date
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            date: Date in YYYY-MM-DD format
            data_dir: Directory containing LOBSTER files
        """
        self.symbol = symbol
        self.date = date
        self.data_dir = Path(data_dir)
        
        # File paths
        date_str = date.replace("-", "-")
        self.message_file = self.data_dir / f"{symbol}_{date}_34200000_57600000_message_1.csv"
        self.orderbook_file = self.data_dir / f"{symbol}_{date}_34200000_57600000_orderbook_1.csv"
        
        # Data containers
        self.messages = []
        self.orderbook_snapshots = []
        
    def parse_messages(self) -> List[ITCHMessage]:
        """Parse message file and return list of ITCH messages"""
        
        if not self.message_file.exists():
            raise FileNotFoundError(f"Message file not found: {self.message_file}")
        
        logger.info(f"Parsing messages from {self.message_file}")
        
        # Read CSV without headers
        df = pd.read_csv(self.message_file, header=None,
                        names=['timestamp', 'type', 'order_id', 'size', 'price', 'direction'])
        
        messages = []
        for _, row in df.iterrows():
            try:
                msg = ITCHMessage(
                    timestamp=row['timestamp'],
                    msg_type=MessageType(int(row['type'])),
                    order_id=int(row['order_id']),
                    size=int(row['size']),
                    price=row['price'] / 10000,  # Convert to dollars
                    direction=int(row['direction'])
                )
                messages.append(msg)
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
        
        self.messages = messages
        logger.info(f"Parsed {len(messages)} messages")
        return messages
    
    def parse_orderbook(self) -> List[OrderBookSnapshot]:
        """Parse orderbook file and return list of snapshots"""
        
        if not self.orderbook_file.exists():
            raise FileNotFoundError(f"Orderbook file not found: {self.orderbook_file}")
        
        logger.info(f"Parsing orderbook from {self.orderbook_file}")
        
        # Read orderbook CSV
        df = pd.read_csv(self.orderbook_file, header=None)
        
        # Get timestamps from messages (orderbook doesn't have timestamps)
        if not self.messages:
            self.parse_messages()
        
        snapshots = []
        for i, row in df.iterrows():
            if i < len(self.messages):
                timestamp = self.messages[i].timestamp
                snapshot = OrderBookSnapshot.from_row(row, timestamp)
                snapshots.append(snapshot)
        
        self.orderbook_snapshots = snapshots
        logger.info(f"Parsed {len(snapshots)} orderbook snapshots")
        return snapshots
    
    def get_trades(self) -> pd.DataFrame:
        """Extract executed trades from messages"""
        
        if not self.messages:
            self.parse_messages()
        
        trades = []
        for msg in self.messages:
            if msg.msg_type in [MessageType.VISIBLE_EXECUTION, MessageType.HIDDEN_EXECUTION]:
                trades.append({
                    'timestamp': msg.timestamp,
                    'datetime': msg.datetime,
                    'symbol': self.symbol,
                    'price': msg.price,
                    'size': msg.size,
                    'side': msg.side,
                    'order_id': msg.order_id
                })
        
        return pd.DataFrame(trades)
    
    def get_order_flow(self) -> pd.DataFrame:
        """Get complete order flow (submissions, cancellations, executions)"""
        
        if not self.messages:
            self.parse_messages()
        
        order_flow = []
        for msg in self.messages:
            order_flow.append({
                'timestamp': msg.timestamp,
                'datetime': msg.datetime,
                'symbol': self.symbol,
                'type': msg.msg_type.name,
                'order_id': msg.order_id,
                'price': msg.price,
                'size': msg.size,
                'side': msg.side
            })
        
        return pd.DataFrame(order_flow)
    
    def get_market_stats(self) -> Dict:
        """Calculate market statistics from the data"""
        
        if not self.orderbook_snapshots:
            self.parse_orderbook()
        
        trades_df = self.get_trades()
        
        stats = {
            'symbol': self.symbol,
            'date': self.date,
            'total_messages': len(self.messages),
            'total_trades': len(trades_df),
            'total_volume': trades_df['size'].sum() if len(trades_df) > 0 else 0,
            'avg_trade_size': trades_df['size'].mean() if len(trades_df) > 0 else 0,
            'price_range': {
                'min': trades_df['price'].min() if len(trades_df) > 0 else 0,
                'max': trades_df['price'].max() if len(trades_df) > 0 else 0,
                'mean': trades_df['price'].mean() if len(trades_df) > 0 else 0
            }
        }
        
        # Calculate spread statistics
        spreads = [s.spread for s in self.orderbook_snapshots if s.spread > 0]
        if spreads:
            stats['spread_stats'] = {
                'mean': np.mean(spreads),
                'median': np.median(spreads),
                'std': np.std(spreads),
                'min': np.min(spreads),
                'max': np.max(spreads)
            }
        
        return stats

class ITCHDataIntegrator:
    """Integrates NASDAQ ITCH data with ABIDES-LLM simulator"""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize integrator
        
        Args:
            data_dir: Directory containing LOBSTER files
        """
        self.data_dir = Path(data_dir)
        self.parsers = {}
        
    def load_symbol(self, symbol: str, date: str) -> LOBSTERDataParser:
        """Load data for a specific symbol and date"""
        
        key = f"{symbol}_{date}"
        if key not in self.parsers:
            self.parsers[key] = LOBSTERDataParser(symbol, date, self.data_dir)
        return self.parsers[key]
    
    def create_replay_events(self, symbol: str, date: str) -> List[Dict]:
        """
        Create replay events for the simulator from ITCH data
        
        Returns list of events that can be fed to the simulator
        """
        
        parser = self.load_symbol(symbol, date)
        messages = parser.parse_messages()
        orderbook = parser.parse_orderbook()
        
        events = []
        for i, msg in enumerate(messages):
            event = {
                'timestamp': msg.datetime,
                'type': 'market_event',
                'symbol': symbol,
                'message_type': msg.msg_type.name,
                'order_id': msg.order_id,
                'price': msg.price,
                'size': msg.size,
                'side': msg.side
            }
            
            # Add orderbook snapshot if available
            if i < len(orderbook):
                snapshot = orderbook[i]
                event['orderbook'] = {
                    'bid': {'price': snapshot.bid_price, 'size': snapshot.bid_size},
                    'ask': {'price': snapshot.ask_price, 'size': snapshot.ask_size},
                    'spread': snapshot.spread,
                    'mid_price': snapshot.mid_price
                }
            
            events.append(event)
        
        return events
    
    def analyze_market_quality(self, symbols: List[str], date: str) -> pd.DataFrame:
        """
        Analyze market quality metrics across multiple symbols
        
        Returns DataFrame with market quality metrics
        """
        
        results = []
        for symbol in symbols:
            try:
                parser = self.load_symbol(symbol, date)
                stats = parser.get_market_stats()
                
                # Flatten nested dictionaries
                flat_stats = {'symbol': symbol, 'date': date}
                flat_stats['total_messages'] = stats['total_messages']
                flat_stats['total_trades'] = stats['total_trades']
                flat_stats['total_volume'] = stats['total_volume']
                flat_stats['avg_trade_size'] = stats['avg_trade_size']
                
                if 'spread_stats' in stats:
                    for key, value in stats['spread_stats'].items():
                        flat_stats[f'spread_{key}'] = value
                
                if 'price_range' in stats:
                    for key, value in stats['price_range'].items():
                        flat_stats[f'price_{key}'] = value
                
                results.append(flat_stats)
                
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def export_for_simulation(self, symbol: str, date: str, output_file: str):
        """
        Export ITCH data in format suitable for ABIDES-LLM simulation
        
        Creates a JSON file with all necessary data for replay
        """
        
        import json
        
        parser = self.load_symbol(symbol, date)
        events = self.create_replay_events(symbol, date)
        stats = parser.get_market_stats()
        
        output = {
            'metadata': {
                'symbol': symbol,
                'date': date,
                'source': 'NASDAQ ITCH via LOBSTER',
                'total_events': len(events),
                'stats': stats
            },
            'events': events
        }
        
        # Convert datetime objects to strings for JSON serialization
        for event in output['events']:
            event['timestamp'] = event['timestamp'].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Exported {len(events)} events to {output_file}")

def main():
    """Example usage of the ITCH data parser"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse AMZN data
    parser = LOBSTERDataParser("AMZN", "2012-06-21", "/workspace")
    
    # Parse messages and orderbook
    messages = parser.parse_messages()
    orderbook = parser.parse_orderbook()
    
    # Get trades
    trades = parser.get_trades()
    print(f"\n📊 AMZN Trading Summary (2012-06-21)")
    print(f"Total messages: {len(messages)}")
    print(f"Total trades: {len(trades)}")
    if len(trades) > 0:
        print(f"Total volume: {trades['size'].sum():,}")
        print(f"Average trade size: {trades['size'].mean():.0f}")
        print(f"Price range: ${trades['price'].min():.2f} - ${trades['price'].max():.2f}")
    
    # Get market statistics
    stats = parser.get_market_stats()
    if 'spread_stats' in stats:
        print(f"\nSpread Statistics:")
        print(f"  Mean: ${stats['spread_stats']['mean']:.4f}")
        print(f"  Median: ${stats['spread_stats']['median']:.4f}")
        print(f"  Std Dev: ${stats['spread_stats']['std']:.4f}")
    
    # Analyze multiple symbols if available
    integrator = ITCHDataIntegrator("/workspace")
    
    # Check which symbols we have
    available_symbols = []
    for symbol in ["AMZN", "AAPL", "MSFT"]:
        if Path(f"/workspace/{symbol}_2012-06-21_34200000_57600000_message_1.csv").exists():
            available_symbols.append(symbol)
    
    if available_symbols:
        print(f"\n📈 Market Quality Analysis")
        print(f"Available symbols: {', '.join(available_symbols)}")
        
        quality_df = integrator.analyze_market_quality(available_symbols, "2012-06-21")
        print("\nMarket Quality Metrics:")
        print(quality_df.to_string())
        
        # Export first symbol for simulation
        export_symbol = available_symbols[0]
        integrator.export_for_simulation(
            export_symbol, 
            "2012-06-21", 
            f"/workspace/{export_symbol}_itch_data.json"
        )
        print(f"\n✅ Exported {export_symbol} data for simulation")

if __name__ == "__main__":
    main()