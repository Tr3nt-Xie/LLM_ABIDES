#!/usr/bin/env python3
"""
LLM-Powered Order Book Analysis System
=====================================

A comprehensive analysis system that uses LLM capabilities to analyze generated
order book data and compare it with real market patterns, providing insights
and recommendations for improving the simulation.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except ImportError:
    print("Warning: seaborn not available. Using default matplotlib style.")
    sns = None
from dataclasses import dataclass
import io
import base64
from real_data_ingestion import fetch_and_compare

# OpenAI integration (v1 API)
try:
    from openai import OpenAI  # v1 client
    from dotenv import load_dotenv
    load_dotenv()
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    _GLOBAL_OPENAI_CLIENT = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else None
except Exception:
    print("Warning: OpenAI not available or not configured. Using mock analysis.")
    _GLOBAL_OPENAI_CLIENT = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataComparison:
    """Results of comparing generated vs real market data"""
    
    # Basic statistics
    generated_stats: Dict[str, Any]
    real_market_benchmarks: Dict[str, Any]
    
    # Comparison metrics
    similarity_scores: Dict[str, float]
    realism_assessment: Dict[str, str]
    
    # Detailed analysis
    microstructure_analysis: Dict[str, Any]
    stylized_facts_compliance: Dict[str, bool]
    
    # Recommendations
    improvement_suggestions: List[str]
    confidence_score: float

class LLMOrderBookAnalyzer:
    """LLM-powered analyzer for order book data quality and realism"""
    
    def __init__(self, use_real_llm: bool = True):
        self.client = _GLOBAL_OPENAI_CLIENT if use_real_llm else None
        self.use_real_llm = self.client is not None
        self.analysis_cache = {}
        
        # Real market benchmarks (typical values for major stocks)
        self.real_market_benchmarks = {
            "bid_ask_spread_bps": {"min": 1, "max": 20, "typical": 5},
            "trade_size_distribution": {
                "small_orders_pct": 0.7,  # < 1000 shares
                "medium_orders_pct": 0.25,  # 1000-10000 shares
                "large_orders_pct": 0.05   # > 10000 shares
            },
            "intraday_volume_pattern": {
                "opening_surge": 2.5,  # volume multiplier
                "midday_quiet": 0.6,
                "closing_surge": 2.0
            },
            "volatility_clustering": True,
            "autocorrelation": {
                "returns": {"1min": 0.05, "5min": 0.02, "hourly": 0.01},
                "volume": {"1min": 0.15, "5min": 0.10, "hourly": 0.05}
            },
            "market_impact": {
                "small_trade_bps": 0.5,
                "large_trade_bps": 5.0
            }
        }
        
        logger.info(f"LLM Analyzer initialized (Real LLM: {self.use_real_llm})")
    
    def analyze_order_book_quality(self, data_dict: Dict[str, pd.DataFrame]) -> MarketDataComparison:
        """Comprehensive analysis of order book data quality and realism"""
        
        logger.info("🔍 Starting comprehensive order book analysis...")
        
        # Extract key datasets
        orders_df = data_dict.get('orders', pd.DataFrame())
        trades_df = data_dict.get('trades', pd.DataFrame())
        snapshots_df = data_dict.get('snapshots', pd.DataFrame())
        
        if orders_df.empty:
            logger.warning("No order data available for analysis")
            return self._create_empty_comparison()
        
        # Generate statistical analysis
        generated_stats = self._calculate_generated_stats(orders_df, trades_df, snapshots_df)
        
        # Compare with real market patterns
        similarity_scores = self._calculate_similarity_scores(generated_stats)
        
        # Assess stylized facts compliance
        stylized_facts = self._assess_stylized_facts(trades_df, snapshots_df)
        
        # Microstructure analysis
        microstructure = self._analyze_microstructure(orders_df, trades_df, snapshots_df)
        
        # Generate LLM-powered insights
        llm_analysis = self._get_llm_insights(generated_stats, similarity_scores, stylized_facts)
        
        # Create comparison object
        comparison = MarketDataComparison(
            generated_stats=generated_stats,
            real_market_benchmarks=self.real_market_benchmarks,
            similarity_scores=similarity_scores,
            realism_assessment=llm_analysis.get("realism_assessment", {}),
            microstructure_analysis=microstructure,
            stylized_facts_compliance=stylized_facts,
            improvement_suggestions=llm_analysis.get("suggestions", []),
            confidence_score=llm_analysis.get("confidence", 0.5)
        )
        
        logger.info("✅ Order book analysis completed")
        return comparison

    def validate_against_real_market(self, data_dict: Dict[str, pd.DataFrame], symbol: str,
                                     start: datetime, end: datetime, interval: str = "1m") -> Dict[str, Any]:
        """Fetch real OHLCV and compare simulated trades against it. Returns a validation package."""
        trades_df = data_dict.get('trades', pd.DataFrame())
        if trades_df.empty:
            return {"error": "No simulated trades to compare"}
        # Filter trades to symbol
        sim_trades = trades_df[trades_df['symbol'] == symbol].copy()
        if sim_trades.empty:
            return {"error": f"No simulated trades for {symbol}"}
        result, used_interval = fetch_and_compare(symbol=symbol, sim_trades=sim_trades, start=start, end=end, interval=interval, allow_fallback=True)
        result["used_interval"] = used_interval
        return result
    
    def _calculate_generated_stats(self, orders_df: pd.DataFrame, trades_df: pd.DataFrame, 
                                 snapshots_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics from generated data"""
        
        stats = {}
        
        # Basic order statistics
        if not orders_df.empty:
            stats["total_orders"] = len(orders_df)
            stats["avg_order_size"] = orders_df['quantity'].mean()
            stats["median_order_size"] = orders_df['quantity'].median()
            stats["order_size_std"] = orders_df['quantity'].std()
            
            # Order type distribution
            stats["market_order_ratio"] = (orders_df['order_type'] == 'MARKET').mean()
            stats["limit_order_ratio"] = (orders_df['order_type'] == 'LIMIT').mean()
            
            # Agent type distribution
            if 'agent_type' in orders_df.columns:
                stats["agent_distribution"] = orders_df['agent_type'].value_counts(normalize=True).to_dict()
            
            # Order size distribution
            small_orders = (orders_df['quantity'] < 1000).mean()
            medium_orders = ((orders_df['quantity'] >= 1000) & (orders_df['quantity'] <= 10000)).mean()
            large_orders = (orders_df['quantity'] > 10000).mean()
            
            stats["order_size_distribution"] = {
                "small_orders_pct": small_orders,
                "medium_orders_pct": medium_orders,
                "large_orders_pct": large_orders
            }
        
        # Trade statistics
        if not trades_df.empty:
            stats["total_trades"] = len(trades_df)
            stats["avg_trade_size"] = trades_df['quantity'].mean()
            stats["total_volume"] = trades_df['quantity'].sum()
            
            # Fill rate
            if not orders_df.empty:
                stats["fill_rate"] = len(trades_df) / len(orders_df)
            
            # Price impact analysis
            if 'market_impact' in trades_df.columns:
                stats["avg_market_impact_bps"] = trades_df['market_impact'].mean() * 10000
                stats["market_impact_std_bps"] = trades_df['market_impact'].std() * 10000
            
            # Volume-weighted average price
            if trades_df['quantity'].sum() > 0:
                stats["vwap"] = (trades_df['price'] * trades_df['quantity']).sum() / trades_df['quantity'].sum()
            
            # Volatility measures
            if len(trades_df) > 1:
                returns = trades_df['price'].pct_change().dropna()
                stats["realized_volatility"] = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized
                stats["return_skewness"] = returns.skew()
                stats["return_kurtosis"] = returns.kurtosis()
        
        # Snapshot statistics (order book depth)
        if not snapshots_df.empty:
            stats["snapshots_count"] = len(snapshots_df)
            
            if 'spread' in snapshots_df.columns:
                # Convert to basis points
                spreads_bps = snapshots_df['spread'] / snapshots_df['last_trade_price'] * 10000
                stats["avg_spread_bps"] = spreads_bps.mean()
                stats["spread_std_bps"] = spreads_bps.std()
            
            if 'mid_price' in snapshots_df.columns:
                mid_returns = snapshots_df['mid_price'].pct_change().dropna()
                if len(mid_returns) > 1:
                    stats["mid_price_volatility"] = mid_returns.std()
        
        # Temporal patterns
        if not orders_df.empty and 'timestamp' in orders_df.columns:
            orders_df['hour'] = pd.to_datetime(orders_df['timestamp']).dt.hour
            hourly_volume = orders_df.groupby('hour')['quantity'].sum()
            
            if len(hourly_volume) > 1:
                stats["intraday_volume_pattern"] = {
                    "opening_volume": hourly_volume.get(9, 0) + hourly_volume.get(10, 0),
                    "midday_volume": hourly_volume.get(12, 0) + hourly_volume.get(13, 0),
                    "closing_volume": hourly_volume.get(15, 0) + hourly_volume.get(16, 0)
                }
        
        return stats
    
    def _calculate_similarity_scores(self, generated_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate similarity scores compared to real market benchmarks"""
        
        scores = {}
        
        # Spread similarity
        if "avg_spread_bps" in generated_stats:
            spread = generated_stats["avg_spread_bps"]
            benchmark_spread = self.real_market_benchmarks["bid_ask_spread_bps"]["typical"]
            spread_diff = abs(spread - benchmark_spread) / benchmark_spread
            scores["spread_similarity"] = max(0, 1 - spread_diff)
        
        # Order size distribution similarity
        if "order_size_distribution" in generated_stats:
            gen_dist = generated_stats["order_size_distribution"]
            bench_dist = self.real_market_benchmarks["trade_size_distribution"]
            
            # Calculate KL divergence (simplified)
            score = 0
            for key in ["small_orders_pct", "medium_orders_pct", "large_orders_pct"]:
                if key in gen_dist and key in bench_dist:
                    gen_val = gen_dist[key]
                    bench_val = bench_dist[key]
                    if bench_val > 0:
                        score += abs(gen_val - bench_val) / bench_val
            
            scores["order_size_similarity"] = max(0, 1 - score / 3)
        
        # Market impact similarity
        if "avg_market_impact_bps" in generated_stats:
            impact = generated_stats["avg_market_impact_bps"]
            # Compare with typical small trade impact
            benchmark_impact = self.real_market_benchmarks["market_impact"]["small_trade_bps"]
            impact_diff = abs(impact - benchmark_impact) / benchmark_impact
            scores["market_impact_similarity"] = max(0, 1 - impact_diff)
        
        # Fill rate realism
        if "fill_rate" in generated_stats:
            fill_rate = generated_stats["fill_rate"]
            # Typical fill rates in active markets are 30-60%
            if 0.3 <= fill_rate <= 0.6:
                scores["fill_rate_realism"] = 1.0
            else:
                scores["fill_rate_realism"] = max(0, 1 - abs(fill_rate - 0.45) / 0.45)
        
        # Overall similarity (average of individual scores)
        if scores:
            scores["overall_similarity"] = np.mean(list(scores.values()))
        
        return scores
    
    def _assess_stylized_facts(self, trades_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> Dict[str, bool]:
        """Assess compliance with market microstructure stylized facts"""
        
        facts = {}
        
        if not trades_df.empty and len(trades_df) > 10:
            # Fact 1: Fat tails in return distribution
            returns = trades_df['price'].pct_change().dropna()
            if len(returns) > 10:
                kurtosis = returns.kurtosis()
                facts["fat_tails"] = kurtosis > 3  # Normal distribution has kurtosis = 3
            
            # Fact 2: Volatility clustering
            if len(returns) > 20:
                # Simple test: autocorrelation in squared returns
                squared_returns = returns ** 2
                autocorr = squared_returns.autocorr(lag=1)
                facts["volatility_clustering"] = autocorr > 0.1
            
            # Fact 3: Mean reversion at high frequencies
            if len(returns) > 5:
                # First-order autocorrelation should be slightly negative
                autocorr_returns = returns.autocorr(lag=1)
                facts["mean_reversion"] = -0.1 < autocorr_returns < 0.05
        
        # Fact 4: Bid-ask spread positivity
        if not snapshots_df.empty and 'spread' in snapshots_df.columns:
            facts["positive_spreads"] = (snapshots_df['spread'] > 0).all()
        
        # Fact 5: Volume patterns
        if not trades_df.empty and 'timestamp' in trades_df.columns:
            if len(trades_df) > 50:
                # Check for intraday volume patterns
                trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
                hourly_volume = trades_df.groupby('hour')['quantity'].sum()
                
                if len(hourly_volume) >= 3:
                    # U-shaped volume pattern (high at open/close, low midday)
                    morning_vol = hourly_volume.iloc[:3].mean() if len(hourly_volume) >= 3 else 0
                    midday_vol = hourly_volume.iloc[3:-3].mean() if len(hourly_volume) >= 7 else morning_vol
                    evening_vol = hourly_volume.iloc[-3:].mean() if len(hourly_volume) >= 3 else 0
                    
                    facts["u_shaped_volume"] = morning_vol > midday_vol and evening_vol > midday_vol
        
        return facts
    
    def _analyze_microstructure(self, orders_df: pd.DataFrame, trades_df: pd.DataFrame, 
                              snapshots_df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed microstructure analysis"""
        
        analysis = {}
        
        # Order flow imbalance
        if not orders_df.empty:
            buy_orders = orders_df[orders_df['side'] == 'BUY']
            sell_orders = orders_df[orders_df['side'] == 'SELL']
            
            analysis["order_flow_imbalance"] = {
                "buy_volume": buy_orders['quantity'].sum(),
                "sell_volume": sell_orders['quantity'].sum(),
                "imbalance_ratio": buy_orders['quantity'].sum() / (buy_orders['quantity'].sum() + sell_orders['quantity'].sum()) if len(orders_df) > 0 else 0.5
            }
        
        # Price impact analysis
        if not trades_df.empty and 'market_impact' in trades_df.columns:
            # Analyze impact by trade size
            trades_df['size_bucket'] = pd.cut(trades_df['quantity'], 
                                            bins=[0, 1000, 10000, float('inf')], 
                                            labels=['small', 'medium', 'large'])
            
            impact_by_size = trades_df.groupby('size_bucket')['market_impact'].mean() * 10000
            analysis["impact_by_trade_size"] = impact_by_size.to_dict()
        
        # Liquidity measures
        if not snapshots_df.empty:
            # Effective spread
            if 'best_bid' in snapshots_df.columns and 'best_ask' in snapshots_df.columns:
                valid_snapshots = snapshots_df.dropna(subset=['best_bid', 'best_ask'])
                if not valid_snapshots.empty:
                    effective_spreads = (valid_snapshots['best_ask'] - valid_snapshots['best_bid']) / valid_snapshots['mid_price']
                    analysis["liquidity_metrics"] = {
                        "avg_effective_spread": effective_spreads.mean(),
                        "spread_volatility": effective_spreads.std()
                    }
        
        return analysis
    
    def _get_llm_insights(self, generated_stats: Dict[str, Any], similarity_scores: Dict[str, float], 
                         stylized_facts: Dict[str, bool]) -> Dict[str, Any]:
        """Use LLM to generate insights and recommendations"""
        
        if not self.use_real_llm:
            return self._get_mock_llm_insights(generated_stats, similarity_scores, stylized_facts)
        
        try:
            # Prepare data for LLM analysis
            analysis_prompt = self._create_analysis_prompt(generated_stats, similarity_scores, stylized_facts)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in financial market microstructure and order book analysis. Provide detailed, technical insights about order book simulation quality."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=700,
                temperature=0.3
            )
            llm_response = response.choices[0].message.content
            return self._parse_llm_response(llm_response)
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}. Using fallback analysis.")
            return self._get_mock_llm_insights(generated_stats, similarity_scores, stylized_facts)
    
    def _create_analysis_prompt(self, generated_stats: Dict[str, Any], similarity_scores: Dict[str, float], 
                              stylized_facts: Dict[str, bool]) -> str:
        """Create a prompt for LLM analysis"""
        
        prompt = f"""
Please analyze the following order book simulation data and provide insights:

GENERATED STATISTICS:
{json.dumps(generated_stats, indent=2, default=str)}

SIMILARITY SCORES (vs real markets):
{json.dumps(similarity_scores, indent=2)}

STYLIZED FACTS COMPLIANCE:
{json.dumps(stylized_facts, indent=2)}

Please provide:
1. Assessment of realism (score 1-10 and explanation)
2. Key strengths of the simulation
3. Main weaknesses and areas for improvement
4. Specific recommendations to improve realism
5. Overall confidence in the simulation quality (0-1)

Format your response as JSON with keys: realism_score, realism_explanation, strengths, weaknesses, suggestions, confidence
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return {
                    "realism_assessment": {
                        "score": parsed.get("realism_score", 5),
                        "explanation": parsed.get("realism_explanation", "")
                    },
                    "suggestions": parsed.get("suggestions", []),
                    "confidence": parsed.get("confidence", 0.5),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", [])
                }
            else:
                # Fallback parsing
                return {
                    "realism_assessment": {"score": 5, "explanation": response[:200]},
                    "suggestions": ["Review LLM response parsing"],
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._get_mock_llm_insights({}, {}, {})
    
    def _get_mock_llm_insights(self, generated_stats: Dict[str, Any], similarity_scores: Dict[str, float], 
                              stylized_facts: Dict[str, bool]) -> Dict[str, Any]:
        """Generate mock insights when LLM is not available"""
        
        # Calculate overall score based on similarity and stylized facts
        overall_similarity = similarity_scores.get("overall_similarity", 0.5)
        stylized_compliance = np.mean(list(stylized_facts.values())) if stylized_facts else 0.5
        
        realism_score = (overall_similarity + stylized_compliance) / 2 * 10
        
        suggestions = []
        
        # Generate suggestions based on weak areas
        if similarity_scores.get("spread_similarity", 1.0) < 0.7:
            suggestions.append("Adjust bid-ask spread generation to match real market spreads more closely")
        
        if similarity_scores.get("order_size_similarity", 1.0) < 0.7:
            suggestions.append("Calibrate order size distribution to better match real trading patterns")
        
        if not stylized_facts.get("volatility_clustering", True):
            suggestions.append("Implement volatility clustering in price generation model")
        
        if not stylized_facts.get("u_shaped_volume", True):
            suggestions.append("Add realistic intraday volume patterns (U-shaped)")
        
        if not suggestions:
            suggestions = ["Overall simulation quality is good", "Consider adding more sophisticated agent behaviors"]
        
        return {
            "realism_assessment": {
                "score": realism_score,
                "explanation": f"Simulation shows {realism_score:.1f}/10 realism based on similarity scores and stylized facts compliance."
            },
            "suggestions": suggestions,
            "confidence": overall_similarity,
            "strengths": ["Basic market structure implemented", "Multiple agent types"],
            "weaknesses": ["Limited sophisticated patterns", "Simplified matching engine"]
        }
    
    def _create_empty_comparison(self) -> MarketDataComparison:
        """Create empty comparison for cases with no data"""
        
        return MarketDataComparison(
            generated_stats={},
            real_market_benchmarks=self.real_market_benchmarks,
            similarity_scores={},
            realism_assessment={"score": 0, "explanation": "No data available"},
            microstructure_analysis={},
            stylized_facts_compliance={},
            improvement_suggestions=["Generate order book data first"],
            confidence_score=0.0
        )
    
    def generate_comparison_report(self, comparison: MarketDataComparison) -> str:
        """Generate a comprehensive comparison report"""
        
        report = []
        report.append("📊 ORDER BOOK QUALITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 30)
        realism_score = comparison.realism_assessment.get("score", 0)
        report.append(f"Overall Realism Score: {realism_score:.1f}/10")
        report.append(f"Confidence Level: {comparison.confidence_score:.1%}")
        report.append("")
        report.append(comparison.realism_assessment.get("explanation", ""))
        report.append("")
        
        # Similarity Analysis
        if comparison.similarity_scores:
            report.append("📈 SIMILARITY TO REAL MARKETS")
            report.append("-" * 30)
            for metric, score in comparison.similarity_scores.items():
                status = "✅" if score > 0.7 else "⚠️" if score > 0.4 else "❌"
                report.append(f"{status} {metric.replace('_', ' ').title()}: {score:.1%}")
            report.append("")
        
        # Stylized Facts Compliance
        if comparison.stylized_facts_compliance:
            report.append("📐 STYLIZED FACTS COMPLIANCE")
            report.append("-" * 30)
            for fact, compliant in comparison.stylized_facts_compliance.items():
                status = "✅" if compliant else "❌"
                report.append(f"{status} {fact.replace('_', ' ').title()}: {'Pass' if compliant else 'Fail'}")
            report.append("")
        
        # Key Statistics
        if comparison.generated_stats:
            report.append("📊 KEY STATISTICS")
            report.append("-" * 30)
            stats = comparison.generated_stats
            
            if "total_orders" in stats:
                report.append(f"Total Orders: {stats['total_orders']:,}")
            if "total_trades" in stats:
                report.append(f"Total Trades: {stats['total_trades']:,}")
            if "fill_rate" in stats:
                report.append(f"Fill Rate: {stats['fill_rate']:.1%}")
            if "avg_spread_bps" in stats:
                report.append(f"Average Spread: {stats['avg_spread_bps']:.1f} bps")
            if "avg_market_impact_bps" in stats:
                report.append(f"Average Market Impact: {stats['avg_market_impact_bps']:.2f} bps")
            report.append("")
        
        # Microstructure Analysis
        if comparison.microstructure_analysis:
            report.append("🔬 MICROSTRUCTURE ANALYSIS")
            report.append("-" * 30)
            
            micro = comparison.microstructure_analysis
            if "order_flow_imbalance" in micro:
                imbalance = micro["order_flow_imbalance"]
                report.append(f"Order Flow Imbalance Ratio: {imbalance.get('imbalance_ratio', 0):.2f}")
            
            if "impact_by_trade_size" in micro:
                report.append("Market Impact by Trade Size:")
                for size, impact in micro["impact_by_trade_size"].items():
                    report.append(f"  {size.title()}: {impact:.2f} bps")
            report.append("")
        
        # Recommendations
        if comparison.improvement_suggestions:
            report.append("💡 IMPROVEMENT RECOMMENDATIONS")
            report.append("-" * 30)
            for i, suggestion in enumerate(comparison.improvement_suggestions, 1):
                report.append(f"{i}. {suggestion}")
            report.append("")
        
        # Benchmarking
        report.append("🎯 REAL MARKET BENCHMARKS")
        report.append("-" * 30)
        benchmarks = comparison.real_market_benchmarks
        report.append(f"Typical Spread: {benchmarks['bid_ask_spread_bps']['typical']} bps")
        report.append(f"Small Order Impact: {benchmarks['market_impact']['small_trade_bps']} bps")
        trade_dist = benchmarks['trade_size_distribution']
        report.append(f"Small Orders: {trade_dist['small_orders_pct']:.0%}")
        report.append(f"Medium Orders: {trade_dist['medium_orders_pct']:.0%}")
        report.append(f"Large Orders: {trade_dist['large_orders_pct']:.0%}")
        report.append("")
        
        report.append("=" * 60)
        report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)
    
    def create_visualization_plots(self, comparison: MarketDataComparison, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create visualization plots and return as base64 encoded strings"""
        
        plots = {}
        
        try:
            # Set style (already set during import)
            pass
            
            # Plot 1: Order size distribution
            if 'orders' in data_dict and not data_dict['orders'].empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                orders_df = data_dict['orders']
                
                ax.hist(orders_df['quantity'], bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Order Size')
                ax.set_ylabel('Frequency')
                ax.set_title('Order Size Distribution')
                ax.set_yscale('log')
                
                plots['order_size_dist'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # Plot 2: Price evolution
            if 'trades' in data_dict and not data_dict['trades'].empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                trades_df = data_dict['trades']
                
                for symbol in trades_df['symbol'].unique():
                    symbol_trades = trades_df[trades_df['symbol'] == symbol]
                    ax.plot(pd.to_datetime(symbol_trades['timestamp']), symbol_trades['price'], 
                           label=symbol, alpha=0.8)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.set_title('Price Evolution by Symbol')
                ax.legend()
                
                plots['price_evolution'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # Plot 3: Similarity scores radar chart
            if comparison.similarity_scores:
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                scores = comparison.similarity_scores
                metrics = list(scores.keys())
                values = list(scores.values())
                
                # Complete the circle
                metrics.append(metrics[0])
                values.append(values[0])
                
                ax.plot(np.linspace(0, 2*np.pi, len(metrics)), values, 'o-', linewidth=2)
                ax.fill(np.linspace(0, 2*np.pi, len(metrics)), values, alpha=0.25)
                ax.set_xticks(np.linspace(0, 2*np.pi, len(metrics)-1))
                ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics[:-1]])
                ax.set_ylim(0, 1)
                ax.set_title('Similarity Scores to Real Markets')
                
                plots['similarity_radar'] = self._fig_to_base64(fig)
                plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
        
        return plots
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')

def main():
    """Main function for testing the LLM analysis system"""
    print("🤖 LLM Order Book Analysis System Test")
    print("=" * 50)
    
    # Create mock data for testing
    mock_orders = pd.DataFrame({
        'order_id': [f'ORD_{i:04d}' for i in range(1000)],
        'timestamp': pd.date_range('2024-01-01 09:30:00', periods=1000, freq='1min'),
        'agent_type': np.random.choice(['retail', 'institutional', 'hft'], 1000),
        'symbol': np.random.choice(['AAPL', 'GOOGL'], 1000),
        'side': np.random.choice(['BUY', 'SELL'], 1000),
        'order_type': np.random.choice(['LIMIT', 'MARKET'], 1000, p=[0.7, 0.3]),
        'price': np.random.normal(100, 5, 1000),
        'quantity': np.random.lognormal(6, 1, 1000).astype(int)
    })
    
    mock_trades = pd.DataFrame({
        'trade_id': [f'TRD_{i:04d}' for i in range(300)],
        'timestamp': pd.date_range('2024-01-01 09:30:00', periods=300, freq='3min'),
        'symbol': np.random.choice(['AAPL', 'GOOGL'], 300),
        'price': np.random.normal(100, 2, 300),
        'quantity': np.random.lognormal(5, 1, 300).astype(int),
        'market_impact': np.random.exponential(0.0001, 300)
    })
    
    data_dict = {
        'orders': mock_orders,
        'trades': mock_trades,
        'snapshots': pd.DataFrame()
    }
    
    # Run analysis
    analyzer = LLMOrderBookAnalyzer()
    comparison = analyzer.analyze_order_book_quality(data_dict)
    
    # Generate report
    report = analyzer.generate_comparison_report(comparison)
    print(report)

if __name__ == "__main__":
    main()