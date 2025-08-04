#!/usr/bin/env python3
"""
LOB Data Analysis and Export Tools
==================================

Comprehensive tools for analyzing and exporting limit order book data
from the scaled LOB database system.

Features:
- High-performance data export to multiple formats
- Statistical analysis of LOB patterns
- Market microstructure metrics calculation
- Order flow analysis and visualization
- Real-time LOB reconstruction from snapshots
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LOBAnalyzer:
    """Comprehensive LOB data analyzer"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Verify database structure
        self._verify_database()
        
        logger.info(f"📊 LOB Analyzer initialized with database: {db_path}")
    
    def _verify_database(self):
        """Verify database has required tables"""
        
        required_tables = ['detailed_orders', 'detailed_trades', 'lob_snapshots']
        
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", 
            self.conn
        )['name'].tolist()
        
        missing_tables = [t for t in required_tables if t not in tables]
        if missing_tables:
            raise ValueError(f"Missing required tables: {missing_tables}")
        
        logger.info(f"✅ Database verified. Tables: {tables}")
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary"""
        
        summary = {}
        
        # Table counts
        for table in ['detailed_orders', 'detailed_trades', 'lob_snapshots']:
            count = pd.read_sql(
                f"SELECT COUNT(*) as count FROM {table}", 
                self.conn
            )['count'].iloc[0]
            summary[f"{table}_count"] = count
        
        # Time range
        time_range = pd.read_sql("""
            SELECT 
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp
            FROM detailed_orders
        """, self.conn)
        summary.update(time_range.iloc[0].to_dict())
        
        # Symbol statistics
        symbol_stats = pd.read_sql("""
            SELECT 
                symbol,
                COUNT(*) as order_count,
                COUNT(DISTINCT agent_id) as unique_agents,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM detailed_orders 
            GROUP BY symbol
            ORDER BY order_count DESC
        """, self.conn)
        summary["symbol_statistics"] = symbol_stats.to_dict('records')
        
        # Agent type distribution
        agent_stats = pd.read_sql("""
            SELECT 
                agent_type,
                COUNT(*) as order_count,
                COUNT(DISTINCT agent_id) as unique_agents,
                AVG(quantity) as avg_order_size,
                SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_orders,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_orders
            FROM detailed_orders 
            GROUP BY agent_type
            ORDER BY order_count DESC
        """, self.conn)
        summary["agent_statistics"] = agent_stats.to_dict('records')
        
        # Trading activity by hour
        hourly_activity = pd.read_sql("""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as order_count,
                COUNT(DISTINCT symbol) as active_symbols
            FROM detailed_orders 
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """, self.conn)
        summary["hourly_activity"] = hourly_activity.to_dict('records')
        
        # Database file size
        db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
        summary["database_size_mb"] = db_size_mb
        
        return summary
    
    def export_orders_data(self, 
                          symbol: Optional[str] = None,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          limit: Optional[int] = None,
                          format: str = 'parquet') -> str:
        """Export orders data with optional filtering"""
        
        # Build query
        where_clauses = []
        if symbol:
            where_clauses.append(f"symbol = '{symbol}'")
        if start_time:
            where_clauses.append(f"timestamp >= '{start_time}'")
        if end_time:
            where_clauses.append(f"timestamp <= '{end_time}'")
        
        where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT 
                order_id, timestamp, agent_id, agent_type, symbol, side, 
                order_type, price, quantity, display_quantity, hidden_quantity,
                status, time_in_force, market_price_at_time, is_aggressive,
                execution_algo, submission_delay_ms
            FROM detailed_orders 
            {where_clause}
            ORDER BY timestamp
            {limit_clause}
        """
        
        # Load data
        logger.info(f"📥 Loading orders data...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df):,} order records")
        
        # Export to specified format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_suffix = f"_{symbol}" if symbol else ""
        
        if format.lower() == 'parquet':
            filename = f"orders_data{symbol_suffix}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
        elif format.lower() == 'csv':
            filename = f"orders_data{symbol_suffix}_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format.lower() == 'hdf5':
            filename = f"orders_data{symbol_suffix}_{timestamp}.h5"
            df.to_hdf(filename, key='orders', mode='w', index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"💾 Exported to: {filename}")
        return filename
    
    def export_lob_snapshots(self, 
                            symbol: Optional[str] = None,
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            include_depth: bool = True,
                            format: str = 'parquet') -> str:
        """Export LOB snapshots with optional depth data"""
        
        # Build query
        where_clauses = []
        if symbol:
            where_clauses.append(f"symbol = '{symbol}'")
        if start_time:
            where_clauses.append(f"timestamp >= '{start_time}'")
        if end_time:
            where_clauses.append(f"timestamp <= '{end_time}'")
        
        where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Select columns based on whether depth data is included
        if include_depth:
            columns = "*"
        else:
            columns = """
                timestamp, symbol, best_bid, best_ask, best_bid_size, best_ask_size,
                absolute_spread, relative_spread_bps, mid_price, weighted_mid_price,
                total_bid_volume, total_ask_volume, volume_imbalance, depth_imbalance,
                price_volatility_1min, volume_rate_1min, trade_count_1min
            """
        
        query = f"""
            SELECT {columns}
            FROM lob_snapshots 
            {where_clause}
            ORDER BY timestamp
        """
        
        # Load data
        logger.info(f"📥 Loading LOB snapshots...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df):,} snapshot records")
        
        # Export to specified format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_suffix = f"_{symbol}" if symbol else ""
        depth_suffix = "_with_depth" if include_depth else ""
        
        if format.lower() == 'parquet':
            filename = f"lob_snapshots{symbol_suffix}{depth_suffix}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
        elif format.lower() == 'csv':
            filename = f"lob_snapshots{symbol_suffix}{depth_suffix}_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format.lower() == 'hdf5':
            filename = f"lob_snapshots{symbol_suffix}{depth_suffix}_{timestamp}.h5"
            df.to_hdf(filename, key='snapshots', mode='w', index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"💾 Exported to: {filename}")
        return filename
    
    def analyze_market_microstructure(self, symbol: str) -> Dict[str, Any]:
        """Analyze market microstructure patterns for a symbol"""
        
        logger.info(f"📊 Analyzing market microstructure for {symbol}")
        
        analysis = {}
        
        # Price impact analysis
        price_impact = pd.read_sql(f"""
            SELECT 
                AVG(market_impact_bps) as avg_market_impact_bps,
                MAX(market_impact_bps) as max_market_impact_bps,
                COUNT(*) as trade_count
            FROM detailed_trades 
            WHERE symbol = '{symbol}'
        """, self.conn)
        analysis["price_impact"] = price_impact.iloc[0].to_dict()
        
        # Spread analysis
        spread_analysis = pd.read_sql(f"""
            SELECT 
                AVG(absolute_spread) as avg_absolute_spread,
                AVG(relative_spread_bps) as avg_relative_spread_bps,
                MIN(relative_spread_bps) as min_spread_bps,
                MAX(relative_spread_bps) as max_spread_bps,
                COUNT(*) as snapshot_count
            FROM lob_snapshots 
            WHERE symbol = '{symbol}'
        """, self.conn)
        analysis["spread_analysis"] = spread_analysis.iloc[0].to_dict()
        
        # Volume analysis
        volume_analysis = pd.read_sql(f"""
            SELECT 
                AVG(total_bid_volume) as avg_bid_volume,
                AVG(total_ask_volume) as avg_ask_volume,
                AVG(volume_imbalance) as avg_volume_imbalance,
                COUNT(*) as observation_count
            FROM lob_snapshots 
            WHERE symbol = '{symbol}'
        """, self.conn)
        analysis["volume_analysis"] = volume_analysis.iloc[0].to_dict()
        
        # Order flow analysis
        order_flow = pd.read_sql(f"""
            SELECT 
                agent_type,
                side,
                COUNT(*) as order_count,
                AVG(quantity) as avg_order_size,
                AVG(price) as avg_price,
                SUM(CASE WHEN is_aggressive THEN 1 ELSE 0 END) as aggressive_orders
            FROM detailed_orders 
            WHERE symbol = '{symbol}'
            GROUP BY agent_type, side
            ORDER BY order_count DESC
        """, self.conn)
        analysis["order_flow"] = order_flow.to_dict('records')
        
        # Intraday patterns
        intraday_patterns = pd.read_sql(f"""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as order_count,
                AVG(quantity) as avg_order_size,
                AVG(price) as avg_price,
                SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_orders,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_orders
            FROM detailed_orders 
            WHERE symbol = '{symbol}'
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """, self.conn)
        analysis["intraday_patterns"] = intraday_patterns.to_dict('records')
        
        return analysis
    
    def reconstruct_lob_at_time(self, symbol: str, timestamp: str, levels: int = 10) -> Dict[str, Any]:
        """Reconstruct order book state at specific timestamp"""
        
        # Get closest snapshot
        snapshot = pd.read_sql(f"""
            SELECT * FROM lob_snapshots 
            WHERE symbol = '{symbol}' 
            AND timestamp <= '{timestamp}'
            ORDER BY timestamp DESC 
            LIMIT 1
        """, self.conn)
        
        if len(snapshot) == 0:
            return {"error": "No snapshot found before the specified time"}
        
        snapshot_data = snapshot.iloc[0]
        
        # Parse depth data
        bid_depth = json.loads(snapshot_data['bid_depth_json'])
        ask_depth = json.loads(snapshot_data['ask_depth_json'])
        
        # Sort and limit levels
        sorted_bids = sorted(
            [(float(price), data) for price, data in bid_depth.items()],
            key=lambda x: x[0], reverse=True
        )[:levels]
        
        sorted_asks = sorted(
            [(float(price), data) for price, data in ask_depth.items()],
            key=lambda x: x[0]
        )[:levels]
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "mid_price": snapshot_data['mid_price'],
            "spread": snapshot_data['absolute_spread'],
            "spread_bps": snapshot_data['relative_spread_bps'],
            "volume_imbalance": snapshot_data['volume_imbalance'],
            "bid_levels": [{"price": price, **data} for price, data in sorted_bids],
            "ask_levels": [{"price": price, **data} for price, data in sorted_asks],
            "best_bid": snapshot_data['best_bid'],
            "best_ask": snapshot_data['best_ask'],
            "total_bid_volume": snapshot_data['total_bid_volume'],
            "total_ask_volume": snapshot_data['total_ask_volume']
        }
    
    def get_execution_quality_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate execution quality metrics"""
        
        # Price improvement analysis
        price_improvement = pd.read_sql(f"""
            SELECT 
                agent_type,
                AVG(CASE WHEN side = 'BUY' THEN market_price_at_time - price ELSE price - market_price_at_time END) as avg_price_improvement,
                COUNT(*) as order_count
            FROM detailed_orders 
            WHERE symbol = '{symbol}' AND order_type = 'LIMIT'
            GROUP BY agent_type
        """, self.conn)
        
        # Fill rate analysis
        fill_rates = pd.read_sql(f"""
            SELECT 
                agent_type,
                COUNT(*) as total_orders,
                COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as filled_orders,
                CAST(COUNT(CASE WHEN status = 'FILLED' THEN 1 END) AS FLOAT) / COUNT(*) as fill_rate
            FROM detailed_orders 
            WHERE symbol = '{symbol}'
            GROUP BY agent_type
        """, self.conn)
        
        # Market impact by trade size
        impact_by_size = pd.read_sql(f"""
            SELECT 
                CASE 
                    WHEN quantity < 1000 THEN 'Small (< 1k)'
                    WHEN quantity < 5000 THEN 'Medium (1k-5k)'
                    WHEN quantity < 10000 THEN 'Large (5k-10k)'
                    ELSE 'Very Large (10k+)'
                END as size_category,
                COUNT(*) as trade_count,
                AVG(market_impact_bps) as avg_impact_bps,
                AVG(permanent_impact_bps) as avg_permanent_impact_bps,
                AVG(temporary_impact_bps) as avg_temporary_impact_bps
            FROM detailed_trades 
            WHERE symbol = '{symbol}'
            GROUP BY size_category
            ORDER BY 
                CASE 
                    WHEN size_category = 'Small (< 1k)' THEN 1
                    WHEN size_category = 'Medium (1k-5k)' THEN 2
                    WHEN size_category = 'Large (5k-10k)' THEN 3
                    ELSE 4
                END
        """, self.conn)
        
        return {
            "price_improvement": price_improvement.to_dict('records'),
            "fill_rates": fill_rates.to_dict('records'),
            "impact_by_size": impact_by_size.to_dict('records')
        }
    
    def export_research_dataset(self, 
                               symbols: Optional[List[str]] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, str]:
        """Export comprehensive research dataset"""
        
        logger.info("📦 Creating comprehensive research dataset...")
        
        files_created = {}
        
        # Filter conditions
        if symbols:
            symbol_list = "','".join(symbols)
            symbol_filter = f"symbol IN ('{symbol_list}')"
        else:
            symbol_filter = "1=1"
        date_filter = ""
        if start_date:
            date_filter += f" AND timestamp >= '{start_date}'"
        if end_date:
            date_filter += f" AND timestamp <= '{end_date}'"
        
        where_clause = f"WHERE {symbol_filter} {date_filter}"
        
        # 1. Orders dataset
        logger.info("📊 Exporting orders dataset...")
        orders_query = f"""
            SELECT 
                order_id, timestamp, agent_id, agent_type, symbol, side, 
                order_type, price, quantity, display_quantity, hidden_quantity,
                status, time_in_force, market_price_at_time, is_aggressive,
                execution_algo, submission_delay_ms
            FROM detailed_orders 
            {where_clause}
            ORDER BY timestamp
        """
        orders_df = pd.read_sql(orders_query, self.conn)
        orders_file = f"research_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        orders_df.to_parquet(orders_file, index=False)
        files_created["orders"] = orders_file
        logger.info(f"✅ Orders: {len(orders_df):,} records -> {orders_file}")
        
        # 2. Trades dataset
        logger.info("📊 Exporting trades dataset...")
        trades_query = f"""
            SELECT 
                trade_id, timestamp, symbol, price, quantity, 
                buy_agent_id, sell_agent_id, aggressor_side,
                market_impact_bps, permanent_impact_bps, temporary_impact_bps,
                pre_trade_mid, post_trade_mid, pre_trade_spread, post_trade_spread,
                matching_latency_microsec
            FROM detailed_trades 
            {where_clause}
            ORDER BY timestamp
        """
        trades_df = pd.read_sql(trades_query, self.conn)
        trades_file = f"research_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        trades_df.to_parquet(trades_file, index=False)
        files_created["trades"] = trades_file
        logger.info(f"✅ Trades: {len(trades_df):,} records -> {trades_file}")
        
        # 3. LOB snapshots (without depth JSON)
        logger.info("📊 Exporting LOB snapshots...")
        snapshots_query = f"""
            SELECT 
                timestamp, symbol, best_bid, best_ask, best_bid_size, best_ask_size,
                absolute_spread, relative_spread_bps, effective_spread_bps,
                mid_price, weighted_mid_price, microprice,
                total_bid_volume, total_ask_volume, 
                bid_volume_5, ask_volume_5, bid_volume_10, ask_volume_10,
                volume_imbalance, depth_imbalance, order_count_imbalance,
                price_volatility_1min, volume_rate_1min, trade_count_1min
            FROM lob_snapshots 
            {where_clause}
            ORDER BY timestamp
        """
        snapshots_df = pd.read_sql(snapshots_query, self.conn)
        snapshots_file = f"research_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        snapshots_df.to_parquet(snapshots_file, index=False)
        files_created["snapshots"] = snapshots_file
        logger.info(f"✅ Snapshots: {len(snapshots_df):,} records -> {snapshots_file}")
        
        # 4. Create metadata file
        metadata = {
            "creation_time": datetime.now().isoformat(),
            "database_source": self.db_path,
            "symbols": symbols,
            "date_range": {"start": start_date, "end": end_date},
            "record_counts": {
                "orders": len(orders_df),
                "trades": len(trades_df),
                "snapshots": len(snapshots_df)
            },
            "files": files_created
        }
        
        metadata_file = f"research_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        files_created["metadata"] = metadata_file
        
        logger.info(f"🎉 Research dataset created: {len(files_created)} files")
        return files_created
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("📊 LOB Analyzer closed")

def main():
    """Example usage of LOB analysis tools"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="LOB Data Analysis Tools")
    parser.add_argument("--db", required=True, help="Database file path")
    parser.add_argument("--summary", action="store_true", help="Generate database summary")
    parser.add_argument("--export-orders", action="store_true", help="Export orders data")
    parser.add_argument("--export-snapshots", action="store_true", help="Export LOB snapshots")
    parser.add_argument("--export-research", action="store_true", help="Export research dataset")
    parser.add_argument("--analyze-symbol", help="Analyze specific symbol microstructure")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "hdf5"], help="Export format")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LOBAnalyzer(args.db)
    
    try:
        if args.summary:
            print("📊 GENERATING DATABASE SUMMARY")
            print("=" * 50)
            summary = analyzer.get_database_summary()
            
            print(f"Database: {args.db}")
            print(f"Size: {summary['database_size_mb']:.1f} MB")
            print(f"Time Range: {summary['first_timestamp']} to {summary['last_timestamp']}")
            print(f"Orders: {summary['detailed_orders_count']:,}")
            print(f"Trades: {summary['detailed_trades_count']:,}")
            print(f"Snapshots: {summary['lob_snapshots_count']:,}")
            
            print("\nSymbol Statistics:")
            for stat in summary['symbol_statistics']:
                print(f"  {stat['symbol']}: {stat['order_count']:,} orders, avg price ${stat['avg_price']:.2f}")
            
            print("\nAgent Type Distribution:")
            for stat in summary['agent_statistics']:
                print(f"  {stat['agent_type']}: {stat['order_count']:,} orders ({stat['unique_agents']} agents)")
        
        if args.export_orders:
            print("📥 EXPORTING ORDERS DATA")
            print("=" * 50)
            filename = analyzer.export_orders_data(symbol=args.symbol, format=args.format)
            print(f"✅ Exported to: {filename}")
        
        if args.export_snapshots:
            print("📥 EXPORTING LOB SNAPSHOTS")
            print("=" * 50)
            filename = analyzer.export_lob_snapshots(symbol=args.symbol, format=args.format)
            print(f"✅ Exported to: {filename}")
        
        if args.export_research:
            print("📦 CREATING RESEARCH DATASET")
            print("=" * 50)
            files = analyzer.export_research_dataset()
            print("✅ Files created:")
            for dataset, filename in files.items():
                print(f"  {dataset}: {filename}")
        
        if args.analyze_symbol:
            print(f"📊 ANALYZING SYMBOL: {args.analyze_symbol}")
            print("=" * 50)
            analysis = analyzer.analyze_market_microstructure(args.analyze_symbol)
            
            print("Price Impact Analysis:")
            impact = analysis['price_impact']
            print(f"  Average Market Impact: {impact['avg_market_impact_bps']:.2f} bps")
            print(f"  Max Market Impact: {impact['max_market_impact_bps']:.2f} bps")
            print(f"  Trade Count: {impact['trade_count']:,}")
            
            print("\nSpread Analysis:")
            spread = analysis['spread_analysis']
            print(f"  Average Spread: {spread['avg_relative_spread_bps']:.2f} bps")
            print(f"  Min/Max Spread: {spread['min_spread_bps']:.2f} / {spread['max_spread_bps']:.2f} bps")
            
            print("\nVolume Analysis:")
            volume = analysis['volume_analysis']
            print(f"  Average Bid Volume: {volume['avg_bid_volume']:,.0f}")
            print(f"  Average Ask Volume: {volume['avg_ask_volume']:,.0f}")
            print(f"  Average Volume Imbalance: {volume['avg_volume_imbalance']:.4f}")
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()