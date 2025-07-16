"""
Enhanced ABIDES Validation System
=================================

Comprehensive validation framework implementing experiments from the ABIDES paper
with enhanced data recording for order books, market microstructure, and stylized facts.

Based on: Byrd, David, et al. "ABIDES: Towards High-Fidelity Market Simulation for AI Research." (2019)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot for detailed analysis"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # (price, volume) pairs
    asks: List[Tuple[float, int]]  # (price, volume) pairs
    last_trade_price: float
    last_trade_volume: int
    spread: float
    mid_price: float
    total_bid_volume: int
    total_ask_volume: int
    depth_levels: int = 10
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'bids': self.bids,
            'asks': self.asks,
            'last_trade_price': self.last_trade_price,
            'last_trade_volume': self.last_trade_volume,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'depth_levels': self.depth_levels
        }


@dataclass
class MarketMicrostructureMetrics:
    """Market microstructure metrics following ABIDES paper methodology"""
    timestamp: datetime
    symbol: str
    
    # Price Metrics
    returns: List[float]
    log_returns: List[float]
    volatility: float
    realized_volatility: float
    
    # Liquidity Metrics
    bid_ask_spread: float
    quoted_spread: float
    effective_spread: float
    price_impact: float
    market_depth: float
    
    # Trading Activity
    trade_frequency: float
    volume_imbalance: float
    order_flow_imbalance: float
    
    # Stylized Facts
    autocorrelation_returns: float
    volatility_clustering: float
    heavy_tails_kurtosis: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EnhancedOrderBookRecorder:
    """Enhanced order book recording system with high-frequency data capture"""
    
    def __init__(self, output_dir: str = "validation_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize databases
        self.db_path = self.output_dir / "market_data.db"
        self._init_database()
        
        # Data storage
        self.order_book_snapshots = []
        self.trade_data = []
        self.market_events = []
        self.agent_actions = []
        
        logger.info(f"Enhanced Order Book Recorder initialized: {self.output_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for market data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Order book snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bids TEXT,
                asks TEXT,
                last_trade_price REAL,
                last_trade_volume INTEGER,
                spread REAL,
                mid_price REAL,
                total_bid_volume INTEGER,
                total_ask_volume INTEGER
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL,
                side TEXT NOT NULL,
                buyer_id TEXT,
                seller_id TEXT,
                trade_type TEXT
            )
        ''')
        
        # Market events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                description TEXT,
                data TEXT
            )
        ''')
        
        # Agent actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                action_type TEXT NOT NULL,
                symbol TEXT,
                price REAL,
                volume INTEGER,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_order_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Record complete order book snapshot"""
        self.order_book_snapshots.append(snapshot)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO order_book_snapshots 
            (timestamp, symbol, bids, asks, last_trade_price, last_trade_volume, 
             spread, mid_price, total_bid_volume, total_ask_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp.isoformat(),
            snapshot.symbol,
            json.dumps(snapshot.bids),
            json.dumps(snapshot.asks),
            snapshot.last_trade_price,
            snapshot.last_trade_volume,
            snapshot.spread,
            snapshot.mid_price,
            snapshot.total_bid_volume,
            snapshot.total_ask_volume
        ))
        
        conn.commit()
        conn.close()
    
    def record_trade(self, timestamp: datetime, symbol: str, price: float, 
                    volume: int, side: str, buyer_id: str = None, 
                    seller_id: str = None, trade_type: str = "market"):
        """Record individual trade"""
        trade_data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'side': side,
            'buyer_id': buyer_id,
            'seller_id': seller_id,
            'trade_type': trade_type
        }
        self.trade_data.append(trade_data)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, price, volume, side, buyer_id, seller_id, trade_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), symbol, price, volume, side, 
            buyer_id, seller_id, trade_type
        ))
        
        conn.commit()
        conn.close()
    
    def record_agent_action(self, timestamp: datetime, agent_id: str, 
                           agent_type: str, action_type: str, symbol: str = None,
                           price: float = None, volume: int = None, 
                           reasoning: str = None):
        """Record agent decision and reasoning"""
        action_data = {
            'timestamp': timestamp,
            'agent_id': agent_id,
            'agent_type': agent_type,
            'action_type': action_type,
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'reasoning': reasoning
        }
        self.agent_actions.append(action_data)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_actions 
            (timestamp, agent_id, agent_type, action_type, symbol, price, volume, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), agent_id, agent_type, action_type,
            symbol, price, volume, reasoning
        ))
        
        conn.commit()
        conn.close()


class StylizedFactsValidator:
    """Validates simulation results against known stylized facts of financial markets"""
    
    def __init__(self, recorder: EnhancedOrderBookRecorder):
        self.recorder = recorder
        self.results = {}
        logger.info("Stylized Facts Validator initialized")
    
    def validate_all_stylized_facts(self, symbol: str) -> Dict[str, Any]:
        """Validate all stylized facts for a given symbol"""
        
        # Get price and return data
        price_data = self._get_price_data(symbol)
        if len(price_data) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(price_data)} points")
            return {}
        
        results = {
            'symbol': symbol,
            'data_points': len(price_data),
            'time_period': {
                'start': price_data.index[0].isoformat(),
                'end': price_data.index[-1].isoformat()
            }
        }
        
        # Stylized Fact 1: Non-normal return distribution
        results['return_distribution'] = self._validate_return_distribution(price_data)
        
        # Stylized Fact 2: Volatility clustering
        results['volatility_clustering'] = self._validate_volatility_clustering(price_data)
        
        # Stylized Fact 3: Heavy tails
        results['heavy_tails'] = self._validate_heavy_tails(price_data)
        
        # Stylized Fact 4: Absence of autocorrelation in returns
        results['return_autocorrelation'] = self._validate_return_autocorrelation(price_data)
        
        # Stylized Fact 5: Long memory in volatility
        results['volatility_persistence'] = self._validate_volatility_persistence(price_data)
        
        # Stylized Fact 6: Leverage effect
        results['leverage_effect'] = self._validate_leverage_effect(price_data)
        
        # Market microstructure facts
        results['microstructure'] = self._validate_microstructure_facts(symbol)
        
        self.results[symbol] = results
        return results
    
    def _get_price_data(self, symbol: str) -> pd.Series:
        """Extract price time series from recorded data"""
        conn = sqlite3.connect(self.recorder.db_path)
        
        query = '''
            SELECT timestamp, mid_price 
            FROM order_book_snapshots 
            WHERE symbol = ? 
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if df.empty:
            return pd.Series()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df['mid_price'].dropna()
    
    def _validate_return_distribution(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate that returns are non-normal with specific characteristics"""
        returns = price_data.pct_change().dropna()
        
        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(returns.values[:5000] if len(returns) > 5000 else returns.values)
        
        # Descriptive statistics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'normality_test': {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            },
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'validation': {
                'non_normal': p_value < 0.05,
                'excess_kurtosis': kurtosis > 0,
                'acceptable_skewness': abs(skewness) < 2
            }
        }
    
    def _validate_volatility_clustering(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate volatility clustering (ARCH effects)"""
        returns = price_data.pct_change().dropna()
        squared_returns = returns ** 2
        
        # Test for autocorrelation in squared returns
        autocorr_1 = squared_returns.autocorr(lag=1)
        autocorr_5 = squared_returns.autocorr(lag=5)
        autocorr_20 = squared_returns.autocorr(lag=20)
        
        # ARCH LM test approximation
        autocorr_significant = autocorr_1 > 0.1
        
        return {
            'autocorrelation_squared_returns': {
                'lag_1': autocorr_1,
                'lag_5': autocorr_5,
                'lag_20': autocorr_20
            },
            'clustering_present': autocorr_significant,
            'validation': {
                'volatility_clustering': autocorr_significant
            }
        }
    
    def _validate_heavy_tails(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate heavy-tailed distribution of returns"""
        returns = price_data.pct_change().dropna()
        
        # Calculate kurtosis
        kurtosis = stats.kurtosis(returns)
        excess_kurtosis = kurtosis
        
        # Tail ratio analysis
        std_dev = returns.std()
        tail_2std = (abs(returns) > 2 * std_dev).sum() / len(returns)
        tail_3std = (abs(returns) > 3 * std_dev).sum() / len(returns)
        
        # Normal distribution expectations
        normal_tail_2std = 2 * (1 - stats.norm.cdf(2))  # ~0.045
        normal_tail_3std = 2 * (1 - stats.norm.cdf(3))  # ~0.003
        
        return {
            'excess_kurtosis': excess_kurtosis,
            'tail_ratios': {
                '2_std': tail_2std,
                '3_std': tail_3std
            },
            'normal_expectations': {
                '2_std': normal_tail_2std,
                '3_std': normal_tail_3std
            },
            'validation': {
                'heavy_tails': excess_kurtosis > 1,
                'fat_tails_2std': tail_2std > normal_tail_2std * 1.5,
                'fat_tails_3std': tail_3std > normal_tail_3std * 2
            }
        }
    
    def _validate_return_autocorrelation(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate absence of autocorrelation in returns"""
        returns = price_data.pct_change().dropna()
        
        # Calculate autocorrelations for various lags
        autocorrs = {}
        for lag in [1, 5, 10, 20]:
            autocorrs[f'lag_{lag}'] = returns.autocorr(lag=lag)
        
        # Test significance (rough approximation)
        n = len(returns)
        significance_threshold = 1.96 / np.sqrt(n)  # 95% confidence
        
        return {
            'autocorrelations': autocorrs,
            'significance_threshold': significance_threshold,
            'validation': {
                'no_significant_autocorr': all(abs(ac) < significance_threshold 
                                             for ac in autocorrs.values())
            }
        }
    
    def _validate_volatility_persistence(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate long memory in volatility"""
        returns = price_data.pct_change().dropna()
        
        # Use rolling volatility as proxy for volatility series
        vol_window = min(20, len(returns) // 10)
        volatility = returns.rolling(window=vol_window).std().dropna()
        
        # Calculate autocorrelations of volatility
        vol_autocorrs = {}
        for lag in [1, 5, 10, 20, 50]:
            if lag < len(volatility):
                vol_autocorrs[f'lag_{lag}'] = volatility.autocorr(lag=lag)
        
        return {
            'volatility_autocorrelations': vol_autocorrs,
            'validation': {
                'persistent_volatility': any(ac > 0.1 for ac in vol_autocorrs.values() 
                                           if not np.isnan(ac))
            }
        }
    
    def _validate_leverage_effect(self, price_data: pd.Series) -> Dict[str, Any]:
        """Validate leverage effect (negative correlation between returns and volatility changes)"""
        returns = price_data.pct_change().dropna()
        
        if len(returns) < 40:
            return {'validation': {'leverage_effect': False}, 'insufficient_data': True}
        
        # Calculate rolling volatility
        vol_window = min(20, len(returns) // 5)
        volatility = returns.rolling(window=vol_window).std()
        vol_changes = volatility.pct_change().dropna()
        
        # Align returns and volatility changes
        common_index = returns.index.intersection(vol_changes.index)
        if len(common_index) < 10:
            return {'validation': {'leverage_effect': False}, 'insufficient_aligned_data': True}
        
        aligned_returns = returns.loc[common_index]
        aligned_vol_changes = vol_changes.loc[common_index]
        
        # Calculate correlation
        correlation = aligned_returns.corr(aligned_vol_changes)
        
        return {
            'return_volatility_correlation': correlation,
            'validation': {
                'leverage_effect': correlation < -0.1
            }
        }
    
    def _validate_microstructure_facts(self, symbol: str) -> Dict[str, Any]:
        """Validate market microstructure stylized facts"""
        conn = sqlite3.connect(self.recorder.db_path)
        
        # Get spread data
        spread_query = '''
            SELECT timestamp, spread, mid_price, total_bid_volume, total_ask_volume
            FROM order_book_snapshots 
            WHERE symbol = ? 
            ORDER BY timestamp
        '''
        
        spread_df = pd.read_sql_query(spread_query, conn, params=(symbol,))
        
        if spread_df.empty:
            conn.close()
            return {'validation': {'sufficient_data': False}}
        
        # Get trade data
        trade_query = '''
            SELECT timestamp, price, volume
            FROM trades 
            WHERE symbol = ? 
            ORDER BY timestamp
        '''
        
        trade_df = pd.read_sql_query(trade_query, conn, params=(symbol,))
        conn.close()
        
        results = {}
        
        # Spread analysis
        if not spread_df['spread'].empty:
            results['spread_stats'] = {
                'mean_spread': spread_df['spread'].mean(),
                'median_spread': spread_df['spread'].median(),
                'spread_volatility': spread_df['spread'].std(),
                'relative_spread': (spread_df['spread'] / spread_df['mid_price']).mean()
            }
        
        # Volume analysis
        if not trade_df.empty:
            results['volume_stats'] = {
                'mean_volume': trade_df['volume'].mean(),
                'median_volume': trade_df['volume'].median(),
                'volume_std': trade_df['volume'].std(),
                'total_volume': trade_df['volume'].sum()
            }
        
        return {
            'microstructure_metrics': results,
            'validation': {
                'sufficient_data': len(spread_df) > 50 and len(trade_df) > 10
            }
        }


class MarketImpactAnalyzer:
    """Analyzes market impact following ABIDES paper methodology"""
    
    def __init__(self, recorder: EnhancedOrderBookRecorder):
        self.recorder = recorder
        self.impact_studies = []
        logger.info("Market Impact Analyzer initialized")
    
    def analyze_trade_impact(self, trade_timestamp: datetime, symbol: str, 
                           trade_size: int, pre_window: int = 60, 
                           post_window: int = 300) -> Dict[str, Any]:
        """Analyze the market impact of a specific trade"""
        
        conn = sqlite3.connect(self.recorder.db_path)
        
        # Get price data around the trade
        start_time = trade_timestamp - timedelta(seconds=pre_window)
        end_time = trade_timestamp + timedelta(seconds=post_window)
        
        query = '''
            SELECT timestamp, mid_price, spread, total_bid_volume, total_ask_volume
            FROM order_book_snapshots 
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(
            symbol, start_time.isoformat(), end_time.isoformat()
        ))
        conn.close()
        
        if df.empty:
            return {'error': 'No data found for impact analysis'}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Find the closest snapshot to trade time
        trade_idx = df.index.get_indexer([trade_timestamp], method='nearest')[0]
        pre_trade_price = df.iloc[trade_idx]['mid_price']
        
        # Calculate price impact over time
        price_impact = (df['mid_price'] - pre_trade_price) / pre_trade_price * 100
        
        # Immediate impact (first 5 seconds)
        immediate_mask = (df.index >= trade_timestamp) & (df.index <= trade_timestamp + timedelta(seconds=5))
        immediate_impact = price_impact[immediate_mask].max() if immediate_mask.any() else 0
        
        # Temporary impact (first 60 seconds)
        temp_mask = (df.index >= trade_timestamp) & (df.index <= trade_timestamp + timedelta(seconds=60))
        temporary_impact = price_impact[temp_mask].max() if temp_mask.any() else 0
        
        # Permanent impact (after 5 minutes)
        perm_mask = df.index >= trade_timestamp + timedelta(seconds=300)
        permanent_impact = price_impact[perm_mask].mean() if perm_mask.any() else 0
        
        impact_study = {
            'trade_timestamp': trade_timestamp.isoformat(),
            'symbol': symbol,
            'trade_size': trade_size,
            'pre_trade_price': pre_trade_price,
            'immediate_impact': immediate_impact,
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'price_impact_series': price_impact.to_dict(),
            'timestamps': [ts.isoformat() for ts in df.index]
        }
        
        self.impact_studies.append(impact_study)
        return impact_study
    
    def generate_impact_report(self) -> Dict[str, Any]:
        """Generate comprehensive market impact report"""
        
        if not self.impact_studies:
            return {'error': 'No impact studies available'}
        
        # Aggregate impact statistics
        immediate_impacts = [study['immediate_impact'] for study in self.impact_studies]
        temporary_impacts = [study['temporary_impact'] for study in self.impact_studies]
        permanent_impacts = [study['permanent_impact'] for study in self.impact_studies]
        
        report = {
            'total_studies': len(self.impact_studies),
            'impact_statistics': {
                'immediate': {
                    'mean': np.mean(immediate_impacts),
                    'median': np.median(immediate_impacts),
                    'std': np.std(immediate_impacts),
                    'max': np.max(immediate_impacts),
                    'min': np.min(immediate_impacts)
                },
                'temporary': {
                    'mean': np.mean(temporary_impacts),
                    'median': np.median(temporary_impacts),
                    'std': np.std(temporary_impacts),
                    'max': np.max(temporary_impacts),
                    'min': np.min(temporary_impacts)
                },
                'permanent': {
                    'mean': np.mean(permanent_impacts),
                    'median': np.median(permanent_impacts),
                    'std': np.std(permanent_impacts),
                    'max': np.max(permanent_impacts),
                    'min': np.min(permanent_impacts)
                }
            },
            'individual_studies': self.impact_studies
        }
        
        return report


class ABIDESValidationFramework:
    """Main validation framework implementing ABIDES paper experiments"""
    
    def __init__(self, output_dir: str = "abides_validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.recorder = EnhancedOrderBookRecorder(str(self.output_dir / "data"))
        self.stylized_facts_validator = StylizedFactsValidator(self.recorder)
        self.impact_analyzer = MarketImpactAnalyzer(self.recorder)
        
        # Results storage
        self.validation_results = {}
        
        logger.info(f"ABIDES Validation Framework initialized: {self.output_dir}")
    
    def run_comprehensive_validation(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive validation following ABIDES paper methodology"""
        
        logger.info("Starting comprehensive ABIDES validation...")
        
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'symbols_analyzed': symbols,
            'framework_version': '1.0',
            'methodology': 'ABIDES Paper Implementation'
        }
        
        # Validate stylized facts for each symbol
        for symbol in symbols:
            logger.info(f"Validating stylized facts for {symbol}")
            stylized_results = self.stylized_facts_validator.validate_all_stylized_facts(symbol)
            results[f'{symbol}_stylized_facts'] = stylized_results
        
        # Generate market impact report
        logger.info("Generating market impact analysis")
        impact_report = self.impact_analyzer.generate_impact_report()
        results['market_impact_analysis'] = impact_report
        
        # Generate summary statistics
        results['summary'] = self._generate_validation_summary(results)
        
        # Save results
        self._save_validation_results(results)
        
        self.validation_results = results
        logger.info("Comprehensive validation completed")
        
        return results
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        
        summary = {
            'total_symbols': len(results['symbols_analyzed']),
            'stylized_facts_compliance': {},
            'data_quality': {},
            'market_realism': {}
        }
        
        # Analyze stylized facts compliance
        fact_counts = {
            'return_distribution': 0,
            'volatility_clustering': 0,
            'heavy_tails': 0,
            'no_autocorrelation': 0,
            'volatility_persistence': 0,
            'leverage_effect': 0
        }
        
        total_symbols = 0
        for key, value in results.items():
            if key.endswith('_stylized_facts') and 'validation' in value:
                total_symbols += 1
                
                if 'return_distribution' in value and 'validation' in value['return_distribution']:
                    val = value['return_distribution']['validation']
                    if val.get('non_normal', False) and val.get('excess_kurtosis', False):
                        fact_counts['return_distribution'] += 1
                
                if 'volatility_clustering' in value and 'validation' in value['volatility_clustering']:
                    if value['volatility_clustering']['validation'].get('volatility_clustering', False):
                        fact_counts['volatility_clustering'] += 1
                
                if 'heavy_tails' in value and 'validation' in value['heavy_tails']:
                    val = value['heavy_tails']['validation']
                    if val.get('heavy_tails', False):
                        fact_counts['heavy_tails'] += 1
                
                if 'return_autocorrelation' in value and 'validation' in value['return_autocorrelation']:
                    if value['return_autocorrelation']['validation'].get('no_significant_autocorr', False):
                        fact_counts['no_autocorrelation'] += 1
                
                if 'volatility_persistence' in value and 'validation' in value['volatility_persistence']:
                    if value['volatility_persistence']['validation'].get('persistent_volatility', False):
                        fact_counts['volatility_persistence'] += 1
                
                if 'leverage_effect' in value and 'validation' in value['leverage_effect']:
                    if value['leverage_effect']['validation'].get('leverage_effect', False):
                        fact_counts['leverage_effect'] += 1
        
        # Calculate compliance rates
        if total_symbols > 0:
            for fact, count in fact_counts.items():
                summary['stylized_facts_compliance'][fact] = count / total_symbols
        
        # Overall compliance score
        compliance_scores = list(summary['stylized_facts_compliance'].values())
        summary['overall_compliance_score'] = np.mean(compliance_scores) if compliance_scores else 0
        
        return summary
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to files"""
        
        # Save main results as JSON
        results_file = self.output_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV
        if 'summary' in results:
            summary_df = pd.DataFrame([results['summary']])
            summary_df.to_csv(self.output_dir / "validation_summary.csv", index=False)
        
        logger.info(f"Validation results saved to {self.output_dir}")
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available. Run validation first."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ABIDES VALIDATION REPORT")
        report_lines.append("=" * 80)
        
        # Summary
        if 'summary' in self.validation_results:
            summary = self.validation_results['summary']
            report_lines.append(f"\nOVERALL COMPLIANCE SCORE: {summary.get('overall_compliance_score', 0):.3f}")
            report_lines.append(f"SYMBOLS ANALYZED: {summary.get('total_symbols', 0)}")
            
            if 'stylized_facts_compliance' in summary:
                report_lines.append("\nSTYLIZED FACTS COMPLIANCE:")
                for fact, rate in summary['stylized_facts_compliance'].items():
                    report_lines.append(f"  {fact.replace('_', ' ').title()}: {rate:.2%}")
        
        # Individual symbol results
        for symbol in self.validation_results.get('symbols_analyzed', []):
            key = f'{symbol}_stylized_facts'
            if key in self.validation_results:
                symbol_results = self.validation_results[key]
                report_lines.append(f"\n{'-' * 50}")
                report_lines.append(f"SYMBOL: {symbol}")
                report_lines.append(f"DATA POINTS: {symbol_results.get('data_points', 0)}")
                
                # Detailed analysis per fact
                facts_to_check = [
                    'return_distribution', 'volatility_clustering', 'heavy_tails',
                    'return_autocorrelation', 'volatility_persistence', 'leverage_effect'
                ]
                
                for fact in facts_to_check:
                    if fact in symbol_results and 'validation' in symbol_results[fact]:
                        validation = symbol_results[fact]['validation']
                        status = "✓" if any(validation.values()) else "✗"
                        report_lines.append(f"  {fact.replace('_', ' ').title()}: {status}")
        
        # Market impact analysis
        if 'market_impact_analysis' in self.validation_results:
            impact = self.validation_results['market_impact_analysis']
            if 'impact_statistics' in impact:
                report_lines.append(f"\n{'-' * 50}")
                report_lines.append("MARKET IMPACT ANALYSIS")
                stats = impact['impact_statistics']
                report_lines.append(f"Studies Conducted: {impact.get('total_studies', 0)}")
                
                for impact_type in ['immediate', 'temporary', 'permanent']:
                    if impact_type in stats:
                        type_stats = stats[impact_type]
                        report_lines.append(f"  {impact_type.title()} Impact:")
                        report_lines.append(f"    Mean: {type_stats.get('mean', 0):.4f}%")
                        report_lines.append(f"    Median: {type_stats.get('median', 0):.4f}%")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.output_dir / "validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return report_text


def create_order_book_snapshot_from_simulation(simulation_data: Dict) -> OrderBookSnapshot:
    """Helper function to create OrderBookSnapshot from simulation data"""
    
    # Extract bid/ask data from simulation
    bids = simulation_data.get('bids', [])
    asks = simulation_data.get('asks', [])
    
    # Calculate metrics
    best_bid = max([bid[0] for bid in bids]) if bids else 0
    best_ask = min([ask[0] for ask in asks]) if asks else 0
    spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
    mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
    
    return OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol=simulation_data.get('symbol', 'UNKNOWN'),
        bids=bids[:10],  # Top 10 levels
        asks=asks[:10],  # Top 10 levels
        last_trade_price=simulation_data.get('last_trade_price', mid_price),
        last_trade_volume=simulation_data.get('last_trade_volume', 0),
        spread=spread,
        mid_price=mid_price,
        total_bid_volume=sum([bid[1] for bid in bids]),
        total_ask_volume=sum([ask[1] for ask in asks])
    )


# Example usage and integration
if __name__ == "__main__":
    # Initialize validation framework
    validator = ABIDESValidationFramework("validation_output")
    
    # Example of how to integrate with existing simulation
    print("ABIDES Enhanced Validation System")
    print("=================================")
    print("\nFramework initialized successfully!")
    print(f"Output directory: {validator.output_dir}")
    print("\nTo use this system:")
    print("1. Record order book snapshots during simulation")
    print("2. Record trades and agent actions")
    print("3. Run comprehensive validation")
    print("4. Generate validation report")
    
    # Example validation (would be called after simulation)
    # results = validator.run_comprehensive_validation(['AAPL', 'MSFT'])
    # report = validator.generate_validation_report()
    # print(report)