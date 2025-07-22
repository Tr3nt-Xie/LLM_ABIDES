#!/usr/bin/env python3
"""
Order Book Data Scaler
======================

System to scale up order book data generation for large-scale analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random
import logging
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScalingConfig:
    """Configuration for scaling order book data"""
    scale_factor: int = 100
    num_symbols: int = 10
    simulation_hours: int = 8
    days_to_simulate: int = 30
    base_orders_per_minute: int = 100
    output_format: str = "csv"
    batch_size: int = 10000
    enable_compression: bool = True

class DataScaler:
    """Main class for scaling order book data"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.symbols = self._generate_symbols()
        self.current_prices = self._initialize_prices()
        self.output_dir = Path("scaled_data")
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            "total_orders": 0,
            "start_time": None,
            "end_time": None
        }
    
    def _generate_symbols(self) -> List[str]:
        """Generate stock symbols"""
        base = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM", "JNJ", "V"]
        if self.config.num_symbols <= len(base):
            return base[:self.config.num_symbols]
        else:
            symbols = base.copy()
            for i in range(self.config.num_symbols - len(base)):
                symbols.append(f"SYM{i+1:03d}")
            return symbols
    
    def _initialize_prices(self) -> Dict[str, float]:
        """Initialize starting prices"""
        prices = {}
        for symbol in self.symbols:
            prices[symbol] = random.uniform(50, 500)
        return prices
    
    def scale_up_data(self) -> Dict[str, Any]:
        """Main scaling method"""
        logger.info("🚀 Starting Order Book Data Scaling")
        logger.info(f"Scale Factor: {self.config.scale_factor}x")
        logger.info(f"Symbols: {len(self.symbols)}")
        
        self.stats["start_time"] = datetime.now()
        
        total_minutes = self.config.days_to_simulate * self.config.simulation_hours * 60
        all_orders = []
        
        for minute in range(total_minutes):
            # Generate orders for this minute
            orders_this_minute = self._generate_minute_orders(minute)
            all_orders.extend(orders_this_minute)
            
            # Save in batches
            if len(all_orders) >= self.config.batch_size:
                self._save_batch(all_orders)
                all_orders = []
            
            if minute % 1000 == 0:
                logger.info(f"Processed {minute:,}/{total_minutes:,} minutes")
        
        # Save remaining orders
        if all_orders:
            self._save_batch(all_orders)
        
        self.stats["end_time"] = datetime.now()
        return self._generate_summary()
    
    def _generate_minute_orders(self, minute: int) -> List[Dict]:
        """Generate orders for one minute"""
        orders = []
        
        for symbol in self.symbols:
            # Update price
            price_change = random.gauss(0, 0.01)  # 1% volatility
            self.current_prices[symbol] *= (1 + price_change)
            self.current_prices[symbol] = max(0.01, self.current_prices[symbol])
            
            current_price = self.current_prices[symbol]
            
            # Generate orders
            num_orders = np.random.poisson(self.config.base_orders_per_minute * self.config.scale_factor / len(self.symbols))
            
            for i in range(num_orders):
                order = self._create_order(symbol, current_price, minute)
                if order:
                    orders.append(order)
                    self.stats["total_orders"] += 1
        
        return orders
    
    def _create_order(self, symbol: str, current_price: float, minute: int) -> Dict:
        """Create a single order"""
        timestamp = datetime(2024, 1, 2, 9, 30) + timedelta(minutes=minute)
        
        agent_type = random.choice(["retail", "institutional", "hft", "market_maker"])
        side = random.choice(["BUY", "SELL"])
        
        # Order size based on agent type
        if agent_type == "retail":
            quantity = random.randint(10, 1000)
        elif agent_type == "institutional":
            quantity = random.randint(1000, 50000)
        elif agent_type == "hft":
            quantity = random.randint(50, 500)
        else:  # market_maker
            quantity = random.randint(100, 5000)
        
        # Price
        spread = current_price * 0.001
        if random.random() < 0.3:  # 30% market orders
            order_type = "MARKET"
            price = current_price
        else:
            order_type = "LIMIT"
            if side == "BUY":
                price = current_price - random.uniform(0, spread * 2)
            else:
                price = current_price + random.uniform(0, spread * 2)
        
        return {
            "timestamp": timestamp,
            "order_id": f"ORD_{minute:08d}_{random.randint(10000, 99999)}",
            "agent_type": agent_type,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "price": round(price, 2),
            "quantity": quantity,
            "market_price": round(current_price, 2)
        }
    
    def _save_batch(self, orders: List[Dict]):
        """Save batch of orders"""
        if not orders:
            return
        
        df = pd.DataFrame(orders)
        batch_num = len(list(self.output_dir.glob("*.csv")))
        
        filename = f"orders_batch_{batch_num:06d}.csv"
        filepath = self.output_dir / filename
        
        compression = 'gzip' if self.config.enable_compression else None
        df.to_csv(filepath, index=False, compression=compression)
        
        logger.info(f"Saved batch {batch_num}: {len(orders):,} orders")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        return {
            "total_orders": self.stats["total_orders"],
            "duration": str(duration),
            "orders_per_second": self.stats["total_orders"] / duration.total_seconds(),
            "symbols": self.symbols,
            "final_prices": self.current_prices,
            "output_directory": str(self.output_dir)
        }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print summary"""
        print("\n" + "="*50)
        print("📊 ORDER BOOK SCALING SUMMARY")
        print("="*50)
        print(f"Total Orders: {summary['total_orders']:,}")
        print(f"Duration: {summary['duration']}")
        print(f"Orders/sec: {summary['orders_per_second']:,.1f}")
        print(f"Symbols: {len(summary['symbols'])}")
        print(f"Output: {summary['output_directory']}")
        print("="*50)

def main():
    """Main function"""
    print("🚀 Order Book Data Scaler")
    
    # Simple configuration
    config = ScalingConfig(
        scale_factor=100,
        num_symbols=10,
        days_to_simulate=7,
        base_orders_per_minute=100
    )
    
    scaler = DataScaler(config)
    summary = scaler.scale_up_data()
    scaler.print_summary(summary)

if __name__ == "__main__":
    main()
