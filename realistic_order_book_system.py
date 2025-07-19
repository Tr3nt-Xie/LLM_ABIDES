"""
Realistic Order Book System
==========================

A comprehensive order book implementation for market simulation with proper
order matching, execution recording, and market microstructure modeling.
"""

import asyncio
import json
import logging
import heapq
import bisect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
import threading
import queue as Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the exchange"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTC = "GTC"  # Good Till Cancel


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Individual order representation"""
    order_id: str
    agent_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_fill_price': self.avg_fill_price
        }


@dataclass
class Trade:
    """Individual trade/execution record"""
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    buyer_agent_id: str
    seller_agent_id: str
    symbol: str
    quantity: int
    price: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'buy_order_id': self.buy_order_id,
            'sell_order_id': self.sell_order_id,
            'buyer_agent_id': self.buyer_agent_id,
            'seller_agent_id': self.seller_agent_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat()
        }


class PriceLevel:
    """A price level in the order book"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: List[Order] = []
        self.total_quantity = 0
    
    def add_order(self, order: Order):
        """Add order to this price level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order: Order):
        """Remove order from this price level"""
        if order in self.orders:
            self.orders.remove(order)
            self.total_quantity -= order.remaining_quantity
    
    def get_total_quantity(self) -> int:
        """Get total quantity at this price level"""
        return sum(order.remaining_quantity for order in self.orders)
    
    def is_empty(self) -> bool:
        """Check if price level is empty"""
        return len(self.orders) == 0


class OrderBook:
    """Complete order book implementation for a single symbol"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, PriceLevel] = {}  # Buy orders
        self.asks: Dict[float, PriceLevel] = {}  # Sell orders
        self.orders: Dict[str, Order] = {}  # All orders by ID
        self.trades: List[Trade] = []
        self.order_sequence = 0
        self.trade_sequence = 0
        
        # Market data
        self.last_trade_price: Optional[float] = None
        self.last_trade_time: Optional[datetime] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.spread: Optional[float] = None
        
        # Statistics
        self.total_volume = 0
        self.trade_count = 0
        self.value_traded = 0.0
    
    def add_order(self, order: Order) -> List[Trade]:
        """Add order to book and return list of executions"""
        trades = []
        
        # Store the order
        self.orders[order.order_id] = order
        
        # Try to match the order
        if order.order_type == OrderType.MARKET:
            trades = self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._execute_limit_order(order)
        
        # Add remaining quantity to book if any
        if order.remaining_quantity > 0 and order.status != OrderStatus.CANCELLED:
            self._add_to_book(order)
        
        # Update market data
        self._update_market_data()
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        # Remove from book
        self._remove_from_book(order)
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        
        self._update_market_data()
        return True
    
    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against existing book"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # Match against asks (lowest prices first)
            prices = sorted(self.asks.keys())
        else:
            # Match against bids (highest prices first)
            prices = sorted(self.bids.keys(), reverse=True)
        
        for price in prices:
            if order.remaining_quantity <= 0:
                break
            
            if order.side == OrderSide.BUY:
                level = self.asks[price]
            else:
                level = self.bids[price]
            
            # Execute against all orders at this level
            level_orders = level.orders.copy()
            for existing_order in level_orders:
                if order.remaining_quantity <= 0:
                    break
                
                trade = self._execute_trade(order, existing_order, price)
                if trade:
                    trades.append(trade)
        
        # Update order status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL
        
        return trades
    
    def _execute_limit_order(self, order: Order) -> List[Trade]:
        """Execute limit order - match what can be matched immediately"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # Can match against asks at or below limit price
            matchable_prices = [p for p in self.asks.keys() if p <= order.price]
            matchable_prices.sort()
        else:
            # Can match against bids at or above limit price
            matchable_prices = [p for p in self.bids.keys() if p >= order.price]
            matchable_prices.sort(reverse=True)
        
        for price in matchable_prices:
            if order.remaining_quantity <= 0:
                break
            
            if order.side == OrderSide.BUY:
                level = self.asks[price]
            else:
                level = self.bids[price]
            
            level_orders = level.orders.copy()
            for existing_order in level_orders:
                if order.remaining_quantity <= 0:
                    break
                
                trade = self._execute_trade(order, existing_order, price)
                if trade:
                    trades.append(trade)
        
        # Update order status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL
        
        return trades
    
    def _execute_trade(self, aggressive_order: Order, passive_order: Order, price: float) -> Optional[Trade]:
        """Execute a trade between two orders"""
        # Determine trade quantity
        trade_quantity = min(aggressive_order.remaining_quantity, passive_order.remaining_quantity)
        
        if trade_quantity <= 0:
            return None
        
        # Create trade record
        self.trade_sequence += 1
        trade = Trade(
            trade_id=f"{self.symbol}_T{self.trade_sequence:06d}",
            buy_order_id=aggressive_order.order_id if aggressive_order.side == OrderSide.BUY else passive_order.order_id,
            sell_order_id=aggressive_order.order_id if aggressive_order.side == OrderSide.SELL else passive_order.order_id,
            buyer_agent_id=aggressive_order.agent_id if aggressive_order.side == OrderSide.BUY else passive_order.agent_id,
            seller_agent_id=aggressive_order.agent_id if aggressive_order.side == OrderSide.SELL else passive_order.agent_id,
            symbol=self.symbol,
            quantity=trade_quantity,
            price=price,
            timestamp=datetime.now()
        )
        
        # Update both orders
        self._update_order_from_trade(aggressive_order, trade_quantity, price)
        self._update_order_from_trade(passive_order, trade_quantity, price)
        
        # Update statistics
        self.trades.append(trade)
        self.total_volume += trade_quantity
        self.trade_count += 1
        self.value_traded += trade_quantity * price
        self.last_trade_price = price
        self.last_trade_time = trade.timestamp
        
        # Remove filled order from book if necessary
        if passive_order.remaining_quantity == 0:
            self._remove_from_book(passive_order)
            passive_order.status = OrderStatus.FILLED
        
        logger.debug(f"Trade executed: {trade_quantity} shares of {self.symbol} at ${price:.2f}")
        
        return trade
    
    def _update_order_from_trade(self, order: Order, quantity: int, price: float):
        """Update order quantities and average price from trade"""
        # Update filled quantity and remaining quantity
        order.filled_quantity += quantity
        order.remaining_quantity -= quantity
        
        # Update average fill price
        total_value = order.avg_fill_price * (order.filled_quantity - quantity) + price * quantity
        order.avg_fill_price = total_value / order.filled_quantity
    
    def _add_to_book(self, order: Order):
        """Add order to the order book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
            self.asks[order.price].add_order(order)
    
    def _remove_from_book(self, order: Order):
        """Remove order from the order book"""
        if order.side == OrderSide.BUY and order.price in self.bids:
            level = self.bids[order.price]
            level.remove_order(order)
            if level.is_empty():
                del self.bids[order.price]
        elif order.side == OrderSide.SELL and order.price in self.asks:
            level = self.asks[order.price]
            level.remove_order(order)
            if level.is_empty():
                del self.asks[order.price]
    
    def _update_market_data(self):
        """Update market data (bid, ask, spread)"""
        # Update best bid
        if self.bids:
            self.bid_price = max(self.bids.keys())
        else:
            self.bid_price = None
        
        # Update best ask
        if self.asks:
            self.ask_price = min(self.asks.keys())
        else:
            self.ask_price = None
        
        # Update spread
        if self.bid_price is not None and self.ask_price is not None:
            self.spread = self.ask_price - self.bid_price
        else:
            self.spread = None
    
    def get_book_snapshot(self, depth: int = 10) -> Dict:
        """Get order book snapshot"""
        # Get top bids (highest prices first)
        bid_prices = sorted(self.bids.keys(), reverse=True)[:depth]
        bids = [(price, self.bids[price].get_total_quantity()) for price in bid_prices]
        
        # Get top asks (lowest prices first)
        ask_prices = sorted(self.asks.keys())[:depth]
        asks = [(price, self.asks[price].get_total_quantity()) for price in ask_prices]
        
        return {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'bids': bids,
            'asks': asks,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.spread,
            'last_trade_price': self.last_trade_price,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'total_volume': self.total_volume,
            'trade_count': self.trade_count,
            'value_traded': self.value_traded
        }
    
    def get_market_stats(self) -> Dict:
        """Get market statistics"""
        if not self.trades:
            return {
                'symbol': self.symbol,
                'total_volume': 0,
                'trade_count': 0,
                'value_traded': 0.0,
                'avg_trade_size': 0,
                'avg_trade_price': 0.0,
                'price_range': None
            }
        
        prices = [trade.price for trade in self.trades]
        quantities = [trade.quantity for trade in self.trades]
        
        return {
            'symbol': self.symbol,
            'total_volume': self.total_volume,
            'trade_count': self.trade_count,
            'value_traded': self.value_traded,
            'avg_trade_size': np.mean(quantities),
            'avg_trade_price': np.mean(prices),
            'price_range': {
                'min': min(prices),
                'max': max(prices),
                'range': max(prices) - min(prices)
            },
            'volume_weighted_avg_price': self.value_traded / self.total_volume if self.total_volume > 0 else 0
        }


class Exchange:
    """Central exchange managing multiple order books"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.order_books: Dict[str, OrderBook] = {}
        self.order_sequence = 0
        self.message_queue = Queue.Queue()
        self.agents: Dict[str, Any] = {}
        
        # Initialize order books
        for symbol in symbols:
            self.order_books[symbol] = OrderBook(symbol)
        
        # Database for persistence
        self.db_path = "market_simulation.db"
        self._setup_database()
        
        logger.info(f"Exchange initialized with {len(symbols)} symbols: {symbols}")
    
    def _setup_database(self):
        """Setup SQLite database for order and trade storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                agent_id TEXT,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                quantity INTEGER,
                price REAL,
                timestamp TEXT,
                status TEXT,
                filled_quantity INTEGER,
                avg_fill_price REAL
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                buy_order_id TEXT,
                sell_order_id TEXT,
                buyer_agent_id TEXT,
                seller_agent_id TEXT,
                symbol TEXT,
                quantity INTEGER,
                price REAL,
                timestamp TEXT
            )
        """)
        
        # Order book snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                snapshot_data TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def submit_order(self, agent_id: str, symbol: str, side: str, order_type: str,
                    quantity: int, price: Optional[float] = None) -> Dict:
        """Submit order to exchange"""
        # Generate order ID
        self.order_sequence += 1
        order_id = f"ORD_{agent_id}_{self.order_sequence:06d}"
        
        # Validate inputs
        if symbol not in self.order_books:
            return {
                'status': 'REJECTED',
                'reason': f'Symbol {symbol} not found',
                'order_id': order_id
            }
        
        try:
            # Create order
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                symbol=symbol,
                side=OrderSide(side.upper()),
                order_type=OrderType(order_type.upper()),
                quantity=quantity,
                price=price
            )
            
            # Submit to order book
            book = self.order_books[symbol]
            trades = book.add_order(order)
            
            # Save to database
            self._save_order(order)
            for trade in trades:
                self._save_trade(trade)
            
            # Notify agents of executions
            for trade in trades:
                self._notify_trade_execution(trade)
            
            return {
                'status': 'ACCEPTED',
                'order_id': order_id,
                'trades': [trade.to_dict() for trade in trades],
                'order_status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'remaining_quantity': order.remaining_quantity
            }
            
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            return {
                'status': 'REJECTED',
                'reason': str(e),
                'order_id': order_id
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        # Find the order
        order = None
        book = None
        
        for symbol, order_book in self.order_books.items():
            if order_id in order_book.orders:
                order = order_book.orders[order_id]
                book = order_book
                break
        
        if not order:
            return {
                'status': 'REJECTED',
                'reason': 'Order not found'
            }
        
        success = book.cancel_order(order_id)
        
        if success:
            self._update_order_in_db(order)
            return {
                'status': 'CANCELLED',
                'order_id': order_id
            }
        else:
            return {
                'status': 'REJECTED',
                'reason': 'Cannot cancel order'
            }
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol"""
        if symbol not in self.order_books:
            return {'error': f'Symbol {symbol} not found'}
        
        return self.order_books[symbol].get_book_snapshot()
    
    def get_market_stats(self, symbol: str) -> Dict:
        """Get market statistics for symbol"""
        if symbol not in self.order_books:
            return {'error': f'Symbol {symbol} not found'}
        
        return self.order_books[symbol].get_market_stats()
    
    def save_order_book_snapshot(self, symbol: str):
        """Save order book snapshot to database"""
        if symbol not in self.order_books:
            return
        
        snapshot = self.order_books[symbol].get_book_snapshot()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO order_book_snapshots (symbol, timestamp, snapshot_data)
            VALUES (?, ?, ?)
        """, (symbol, datetime.now().isoformat(), json.dumps(snapshot)))
        
        conn.commit()
        conn.close()
    
    def _save_order(self, order: Order):
        """Save order to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO orders 
            (order_id, agent_id, symbol, side, order_type, quantity, price, 
             timestamp, status, filled_quantity, avg_fill_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id, order.agent_id, order.symbol, order.side.value,
            order.order_type.value, order.quantity, order.price,
            order.timestamp.isoformat(), order.status.value,
            order.filled_quantity, order.avg_fill_price
        ))
        
        conn.commit()
        conn.close()
    
    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades 
            (trade_id, buy_order_id, sell_order_id, buyer_agent_id, 
             seller_agent_id, symbol, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id, trade.buy_order_id, trade.sell_order_id,
            trade.buyer_agent_id, trade.seller_agent_id, trade.symbol,
            trade.quantity, trade.price, trade.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_order_in_db(self, order: Order):
        """Update order status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE orders 
            SET status = ?, filled_quantity = ?, avg_fill_price = ?
            WHERE order_id = ?
        """, (order.status.value, order.filled_quantity, order.avg_fill_price, order.order_id))
        
        conn.commit()
        conn.close()
    
    def _notify_trade_execution(self, trade: Trade):
        """Notify agents of trade execution"""
        # In a real implementation, this would send messages to agents
        logger.info(f"Trade executed: {trade.quantity} {trade.symbol} at ${trade.price:.2f}")
    
    def get_trading_history(self, symbol: str = None, agent_id: str = None) -> pd.DataFrame:
        """Get trading history as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if agent_id:
            query += " AND (buyer_agent_id = ? OR seller_agent_id = ?)"
            params.extend([agent_id, agent_id])
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def generate_market_report(self) -> Dict:
        """Generate comprehensive market report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'exchange_stats': {
                'total_symbols': len(self.symbols),
                'total_orders': self.order_sequence,
                'active_orders': 0,
                'total_trades': 0,
                'total_volume': 0,
                'total_value': 0.0
            }
        }
        
        for symbol in self.symbols:
            book = self.order_books[symbol]
            stats = book.get_market_stats()
            snapshot = book.get_book_snapshot()
            
            report['symbols'][symbol] = {
                'stats': stats,
                'snapshot': snapshot
            }
            
            # Update exchange totals
            report['exchange_stats']['total_trades'] += stats['trade_count']
            report['exchange_stats']['total_volume'] += stats['total_volume']
            report['exchange_stats']['total_value'] += stats['value_traded']
            report['exchange_stats']['active_orders'] += len(book.orders)
        
        return report