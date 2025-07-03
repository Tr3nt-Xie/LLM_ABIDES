"""
Enhanced LLM-ABIDES Integration System
=====================================

Complete system for realistic trading market simulation combining LLM reasoning 
with ABIDES market microstructure simulation.
"""

import asyncio
import json
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import autogen
from abc import ABC, abstractmethod
import threading
import queue
import time
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCategory(Enum):
    """Categories of news events that can affect market sentiment"""
    EARNINGS = "earnings"
    MERGERS = "mergers" 
    REGULATORY = "regulatory"
    MACRO_ECONOMIC = "macro_economic"
    COMPANY_SPECIFIC = "company_specific"
    GEOPOLITICAL = "geopolitical"
    TECHNICAL = "technical"
    FDA_APPROVAL = "fda_approval"
    ANALYST_UPGRADE = "analyst_upgrade"
    INSIDER_TRADING = "insider_trading"


class MarketSentiment(Enum):
    """Market sentiment levels"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class NewsEvent:
    """Enhanced news event structure with realistic market impact modeling"""
    timestamp: datetime
    category: NewsCategory
    headline: str
    content: str
    affected_symbols: List[str]
    sentiment_score: float  # -1 to 1
    importance: float  # 0 to 1
    source: str = "MarketNews"
    impact_duration: int = 60  # minutes
    sector_impact: Dict[str, float] = None
    confidence: float = 0.8
    related_events: List[str] = None
    
    def __post_init__(self):
        if self.sector_impact is None:
            self.sector_impact = {}
        if self.related_events is None:
            self.related_events = []
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'headline': self.headline,
            'content': self.content,
            'affected_symbols': self.affected_symbols,
            'sentiment_score': self.sentiment_score,
            'importance': self.importance,
            'source': self.source,
            'impact_duration': self.impact_duration,
            'sector_impact': self.sector_impact,
            'confidence': self.confidence,
            'related_events': self.related_events
        }


@dataclass
class MarketSignal:
    """Enhanced market signal with risk management features"""
    timestamp: datetime
    signal_type: str  # 'momentum', 'mean_reversion', 'volatility', 'arbitrage'
    symbol: str
    strength: float  # -1 to 1
    duration: int  # minutes
    confidence: float  # 0 to 1
    source_agent: str
    risk_level: str = "medium"  # low, medium, high
    expected_return: float = 0.0
    max_position: float = 0.1  # max % of portfolio
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.15  # 15% take profit
    sector: str = "unknown"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RealisticNewsGenerator:
    """Enhanced news generator with sector correlation and realistic timing"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.sectors = self._initialize_sectors()
        self.news_templates = self._load_enhanced_templates()
        self.recent_events = []
        self.market_cycle = "normal"  # bull, bear, normal, volatile
        
    def _initialize_sectors(self) -> Dict[str, str]:
        """Map symbols to sectors for realistic correlation"""
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'JPM': 'Financial',
            'BAC': 'Financial',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'XOM': 'Energy'
        }
        return {symbol: sector_mapping.get(symbol, 'Other') for symbol in self.symbols}
    
    def _load_enhanced_templates(self) -> Dict[NewsCategory, List[Dict]]:
        """Load enhanced news templates with realistic parameters"""
        return {
            NewsCategory.EARNINGS: [
                {
                    "template": "{symbol} reports Q{quarter} earnings beating expectations by {percentage}%",
                    "sentiment_range": (0.3, 0.8),
                    "importance_range": (0.6, 0.9),
                    "duration_range": (30, 180),
                    "sector_spillover": 0.3
                },
                {
                    "template": "{symbol} misses earnings forecast, revenue down {percentage}%",
                    "sentiment_range": (-0.8, -0.3),
                    "importance_range": (0.5, 0.8),
                    "duration_range": (60, 240),
                    "sector_spillover": 0.4
                }
            ],
            NewsCategory.MERGERS: [
                {
                    "template": "{symbol} announces ${amount}B acquisition of {target}",
                    "sentiment_range": (0.2, 0.6),
                    "importance_range": (0.7, 1.0),
                    "duration_range": (120, 480),
                    "sector_spillover": 0.5
                }
            ],
            NewsCategory.REGULATORY: [
                {
                    "template": "SEC announces investigation into {symbol} trading practices",
                    "sentiment_range": (-0.7, -0.2),
                    "importance_range": (0.6, 0.9),
                    "duration_range": (240, 720),
                    "sector_spillover": 0.6
                }
            ],
            NewsCategory.MACRO_ECONOMIC: [
                {
                    "template": "Federal Reserve {action} interest rates by {rate}bps",
                    "sentiment_range": (-0.5, 0.5),
                    "importance_range": (0.8, 1.0),
                    "duration_range": (180, 600),
                    "sector_spillover": 0.9
                }
            ]
        }
    
    def generate_realistic_event(self, category: NewsCategory = None, 
                                forced_symbol: str = None) -> NewsEvent:
        """Generate realistic news event with proper market dynamics"""
        
        if category is None:
            # Weight categories by market cycle
            weights = self._get_category_weights()
            category = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        
        templates = self.news_templates[category]
        template_data = random.choice(templates)
        
        # Select symbol with sector correlation consideration
        if forced_symbol:
            symbol = forced_symbol
        else:
            symbol = self._select_symbol_with_correlation(category)
        
        # Generate parameters
        headline = self._generate_headline(template_data, symbol)
        sentiment = random.uniform(*template_data["sentiment_range"])
        importance = random.uniform(*template_data["importance_range"])
        duration = random.randint(*template_data["duration_range"])
        
        # Calculate sector impact
        sector_impact = self._calculate_sector_impact(
            symbol, template_data["sector_spillover"], sentiment
        )
        
        # Determine affected symbols
        affected_symbols = self._get_affected_symbols(symbol, sector_impact)
        
        return NewsEvent(
            timestamp=datetime.now(),
            category=category,
            headline=headline,
            content=f"Full story: {headline}. Market impact expected across {len(affected_symbols)} securities.",
            affected_symbols=affected_symbols,
            sentiment_score=sentiment,
            importance=importance,
            impact_duration=duration,
            sector_impact=sector_impact,
            confidence=random.uniform(0.7, 0.95)
        )
    
    def _get_category_weights(self) -> Dict[NewsCategory, float]:
        """Get category weights based on market cycle"""
        base_weights = {
            NewsCategory.EARNINGS: 0.25,
            NewsCategory.COMPANY_SPECIFIC: 0.20,
            NewsCategory.MACRO_ECONOMIC: 0.15,
            NewsCategory.REGULATORY: 0.10,
            NewsCategory.MERGERS: 0.10,
            NewsCategory.TECHNICAL: 0.10,
            NewsCategory.GEOPOLITICAL: 0.05,
            NewsCategory.FDA_APPROVAL: 0.03,
            NewsCategory.ANALYST_UPGRADE: 0.02
        }
        
        # Adjust weights based on market cycle
        if self.market_cycle == "volatile":
            base_weights[NewsCategory.MACRO_ECONOMIC] *= 2
            base_weights[NewsCategory.GEOPOLITICAL] *= 3
        elif self.market_cycle == "bull":
            base_weights[NewsCategory.EARNINGS] *= 1.5
            base_weights[NewsCategory.MERGERS] *= 1.5
        
        return base_weights
    
    def _select_symbol_with_correlation(self, category: NewsCategory) -> str:
        """Select symbol considering sector correlations"""
        if category == NewsCategory.MACRO_ECONOMIC:
            return random.choice(self.symbols)  # Macro affects all
        
        # Weight by recent event history to avoid clustering
        weights = [1.0] * len(self.symbols)
        for i, symbol in enumerate(self.symbols):
            recent_count = sum(1 for event in self.recent_events[-10:] if symbol in event)
            weights[i] = max(0.1, 1.0 - recent_count * 0.2)
        
        return random.choices(self.symbols, weights=weights)[0]
    
    def _generate_headline(self, template_data: Dict, symbol: str) -> str:
        """Generate realistic headline from template"""
        template = template_data["template"]
        
        # Generate realistic parameters
        percentage = random.randint(1, 25)
        quarter = random.randint(1, 4)
        amount = random.randint(1, 100)
        rate = random.choice([25, 50, 75, 100])
        action = random.choice(["raises", "cuts", "maintains"])
        target = random.choice(["competitor", "startup", "division"])
        
        return template.format(
            symbol=symbol,
            percentage=percentage,
            quarter=quarter,
            amount=amount,
            rate=rate,
            action=action,
            target=target
        )
    
    def _calculate_sector_impact(self, primary_symbol: str, spillover: float, 
                                sentiment: float) -> Dict[str, float]:
        """Calculate realistic sector spillover effects"""
        sector_impact = {}
        primary_sector = self.sectors[primary_symbol]
        
        for symbol, sector in self.sectors.items():
            if sector == primary_sector:
                # Same sector gets full impact
                impact = sentiment * random.uniform(0.7, 1.0)
            else:
                # Cross-sector spillover
                cross_sector_correlation = {
                    ('Technology', 'Technology'): 0.8,
                    ('Financial', 'Financial'): 0.7,
                    ('Healthcare', 'Healthcare'): 0.6,
                    ('Technology', 'Financial'): 0.3,
                    ('Financial', 'Technology'): 0.3,
                }
                correlation = cross_sector_correlation.get((primary_sector, sector), 0.1)
                impact = sentiment * spillover * correlation * random.uniform(0.5, 1.5)
            
            sector_impact[symbol] = max(-1.0, min(1.0, impact))
        
        return sector_impact
    
    def _get_affected_symbols(self, primary_symbol: str, 
                             sector_impact: Dict[str, float]) -> List[str]:
        """Determine which symbols are meaningfully affected"""
        affected = [primary_symbol]  # Primary symbol always affected
        
        for symbol, impact in sector_impact.items():
            if symbol != primary_symbol and abs(impact) > 0.1:
                affected.append(symbol)
        
        return affected


class EnhancedLLMNewsAnalyzer(autogen.ConversableAgent):
    """Enhanced LLM news analyzer with multi-step reasoning and validation"""
    
    def __init__(self, name: str = "EnhancedNewsAnalyzer", **kwargs):
        super().__init__(name, **kwargs)
        self.analysis_history = []
        self.market_context = {}
        self.sector_knowledge = {}
        
    def analyze_news_comprehensive(self, news_event: NewsEvent, 
                                 market_context: Dict = None) -> Dict[str, Any]:
        """Comprehensive news analysis with market context"""
        
        if market_context:
            self.market_context = market_context
        
        # Multi-step analysis prompt
        analysis_prompt = self._create_analysis_prompt(news_event)
        
        try:
            response = self.generate_reply([{"role": "user", "content": analysis_prompt}])
            analysis = self._parse_comprehensive_response(response)
            
            # Validate and enhance analysis
            analysis = self._validate_and_enhance_analysis(analysis, news_event)
            
            # Store for learning
            analysis['news_event'] = news_event.to_dict()
            analysis['timestamp'] = datetime.now().isoformat()
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive news analysis: {e}")
            return self._fallback_analysis(news_event)
    
    def _create_analysis_prompt(self, news_event: NewsEvent) -> str:
        """Create comprehensive analysis prompt with market context"""
        
        market_context_str = ""
        if self.market_context:
            market_context_str = f"""
            Current Market Context:
            - Market volatility: {self.market_context.get('volatility', 'unknown')}
            - Recent trend: {self.market_context.get('trend', 'unknown')}
            - Trading volume: {self.market_context.get('volume', 'unknown')}
            - Sector performance: {self.market_context.get('sector_performance', {})}
            """
        
        return f"""
        As a senior financial analyst, provide a comprehensive analysis of this news event:
        
        News Event:
        - Headline: {news_event.headline}
        - Category: {news_event.category.value}
        - Content: {news_event.content}
        - Affected Symbols: {', '.join(news_event.affected_symbols)}
        - Initial Sentiment: {news_event.sentiment_score}
        
        {market_context_str}
        
        Please provide a detailed analysis in JSON format with these fields:
        
        1. "impact_assessment": {{
            "immediate_impact": (1-10 scale),
            "short_term_impact": (1-10 scale),  
            "long_term_impact": (1-10 scale),
            "market_wide_effect": (1-10 scale)
        }}
        
        2. "price_predictions": {{
            "direction": "up/down/neutral",
            "magnitude_percent": (expected % change),
            "confidence": (0-1),
            "time_horizon": "immediate/hours/days/weeks"
        }}
        
        3. "trading_signals": {{
            "signal_type": "momentum/mean_reversion/volatility/arbitrage",
            "signal_strength": (-1 to 1),
            "risk_level": "low/medium/high",
            "position_sizing": (0-1, max % of portfolio),
            "stop_loss": (% below entry),
            "take_profit": (% above entry)
        }}
        
        4. "sector_analysis": {{
            "primary_sector_impact": (sector name and impact -1 to 1),
            "spillover_sectors": [list of affected sectors],
            "correlation_strength": (0-1)
        }}
        
        5. "risk_factors": [list of key risks]
        
        6. "confidence_factors": {{
            "news_reliability": (0-1),
            "market_timing": (0-1),
            "sector_knowledge": (0-1),
            "overall_confidence": (0-1)
        }}
        
        7. "reasoning": "Detailed explanation of your analysis"
        
        Respond only with valid JSON.
        """
    
    def _parse_comprehensive_response(self, response: str) -> Dict[str, Any]:
        """Parse comprehensive LLM response with fallback handling"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing with regex/heuristics
        return self._heuristic_parsing(response)
    
    def _heuristic_parsing(self, response: str) -> Dict[str, Any]:
        """Heuristic parsing when JSON parsing fails"""
        analysis = {
            "impact_assessment": {
                "immediate_impact": random.randint(4, 8),
                "short_term_impact": random.randint(3, 7),
                "long_term_impact": random.randint(2, 6),
                "market_wide_effect": random.randint(2, 5)
            },
            "price_predictions": {
                "direction": random.choice(["up", "down", "neutral"]),
                "magnitude_percent": random.uniform(0.5, 5.0),
                "confidence": random.uniform(0.5, 0.8),
                "time_horizon": random.choice(["immediate", "hours", "days"])
            },
            "trading_signals": {
                "signal_type": random.choice(["momentum", "mean_reversion", "volatility"]),
                "signal_strength": random.uniform(-0.8, 0.8),
                "risk_level": random.choice(["low", "medium", "high"]),
                "position_sizing": random.uniform(0.05, 0.2),
                "stop_loss": random.uniform(0.02, 0.1),
                "take_profit": random.uniform(0.05, 0.3)
            },
            "reasoning": "Heuristic analysis due to parsing limitations"
        }
        return analysis
    
    def _validate_and_enhance_analysis(self, analysis: Dict, 
                                     news_event: NewsEvent) -> Dict:
        """Validate and enhance analysis with sanity checks"""
        
        # Sanity checks and corrections
        if 'price_predictions' in analysis:
            # Ensure magnitude is reasonable
            magnitude = analysis['price_predictions'].get('magnitude_percent', 0)
            if magnitude > 20:  # Cap at 20% for single news event
                analysis['price_predictions']['magnitude_percent'] = 20
            
            # Ensure direction aligns with sentiment
            direction = analysis['price_predictions'].get('direction', 'neutral')
            if news_event.sentiment_score > 0.3 and direction == 'down':
                analysis['price_predictions']['direction'] = 'up'
            elif news_event.sentiment_score < -0.3 and direction == 'up':
                analysis['price_predictions']['direction'] = 'down'
        
        # Add risk adjustments based on market conditions
        if 'trading_signals' in analysis:
            # Reduce position sizing in high volatility
            volatility = self.market_context.get('volatility', 'medium')
            if volatility == 'high':
                current_sizing = analysis['trading_signals'].get('position_sizing', 0.1)
                analysis['trading_signals']['position_sizing'] = current_sizing * 0.7
        
        return analysis
    
    def _fallback_analysis(self, news_event: NewsEvent) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        direction = 'up' if news_event.sentiment_score > 0 else 'down' if news_event.sentiment_score < 0 else 'neutral'
        
        return {
            'impact_assessment': {
                'immediate_impact': int(abs(news_event.sentiment_score) * 10),
                'short_term_impact': int(abs(news_event.sentiment_score) * 8),
                'long_term_impact': int(abs(news_event.sentiment_score) * 5),
                'market_wide_effect': int(news_event.importance * 10)
            },
            'price_predictions': {
                'direction': direction,
                'magnitude_percent': abs(news_event.sentiment_score) * 5,
                'confidence': news_event.confidence,
                'time_horizon': 'hours'
            },
            'trading_signals': {
                'signal_type': 'momentum',
                'signal_strength': news_event.sentiment_score,
                'risk_level': 'medium',
                'position_sizing': min(0.2, abs(news_event.sentiment_score) * 0.3),
                'stop_loss': 0.05,
                'take_profit': 0.15
            },
            'reasoning': f'Fallback analysis based on sentiment: {news_event.sentiment_score}'
        }


class AdvancedLLMTradingAgent(autogen.ConversableAgent):
    """Advanced LLM trading agent with sophisticated strategy and risk management"""
    
    def __init__(self, name: str, strategy: str, risk_tolerance: float = 0.5, 
                 specialization: str = "generalist", **kwargs):
        super().__init__(name, **kwargs)
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.specialization = specialization  # sector, news_type, or signal_type
        
        # Enhanced portfolio tracking
        self.portfolio = {
            'cash': 1000000,  # $1M starting capital
            'positions': {},
            'open_orders': {},
            'trade_history': [],
            'pnl_history': [],
            'risk_metrics': {},
            'performance_stats': {}
        }
        
        # Strategy parameters
        self.strategy_params = self._initialize_strategy_params()
        
        # Learning and adaptation
        self.performance_tracker = PerformanceTracker()
        self.market_memory = MarketMemory(capacity=1000)
        
    def _initialize_strategy_params(self) -> Dict:
        """Initialize strategy-specific parameters"""
        base_params = {
            'max_position_size': 0.15,  # 15% of portfolio per position
            'max_total_exposure': 0.8,   # 80% max total exposure
            'stop_loss_pct': 0.05,       # 5% stop loss
            'take_profit_pct': 0.20,     # 20% take profit
            'min_confidence': 0.6,       # Minimum signal confidence
            'position_decay': 0.95       # Daily position decay factor
        }
        
        # Strategy-specific adjustments
        strategy_adjustments = {
            'momentum': {
                'max_position_size': 0.20,
                'stop_loss_pct': 0.03,
                'min_confidence': 0.7
            },
            'value': {
                'max_position_size': 0.25,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.30,
                'min_confidence': 0.5
            },
            'volatility': {
                'max_position_size': 0.10,
                'stop_loss_pct': 0.04,
                'min_confidence': 0.8
            },
            'arbitrage': {
                'max_position_size': 0.30,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.10,
                'min_confidence': 0.9
            }
        }
        
        if self.strategy in strategy_adjustments:
            base_params.update(strategy_adjustments[self.strategy])
        
        # Risk tolerance adjustments
        risk_multiplier = 0.5 + self.risk_tolerance
        base_params['max_position_size'] *= risk_multiplier
        base_params['max_total_exposure'] *= risk_multiplier
        
        return base_params
    
    def process_news_with_specialization(self, news_event: NewsEvent, 
                                       analysis: Dict, market_data: Dict) -> List[MarketSignal]:
        """Process news with agent specialization consideration"""
        
        # Check if this agent should respond to this news
        if not self._should_respond_to_news(news_event):
            return []
        
        # Generate base signals
        signals = []
        for symbol in news_event.affected_symbols:
            if symbol in market_data:
                signal = self._generate_enhanced_signal(symbol, analysis, news_event, market_data[symbol])
                if signal and self._validate_signal(signal):
                    signals.append(signal)
        
        return signals
    
    def _should_respond_to_news(self, news_event: NewsEvent) -> bool:
        """Determine if agent should respond based on specialization"""
        
        # Specialization filters
        if self.specialization.startswith('sector_'):
            target_sector = self.specialization.replace('sector_', '')
            # Only respond to news affecting this sector
            return any(symbol for symbol in news_event.affected_symbols 
                      if self._get_symbol_sector(symbol) == target_sector)
        
        elif self.specialization.startswith('news_'):
            target_category = self.specialization.replace('news_', '')
            return news_event.category.value == target_category
        
        elif self.specialization == 'high_frequency':
            # Only respond to high-impact, short-duration news
            return news_event.importance > 0.7 and news_event.impact_duration < 60
        
        elif self.specialization == 'fundamental':
            # Focus on earnings, mergers, regulatory news
            return news_event.category in [NewsCategory.EARNINGS, NewsCategory.MERGERS, NewsCategory.REGULATORY]
        
        # Generalist responds to all news
        return True
    
    def _generate_enhanced_signal(self, symbol: str, analysis: Dict, 
                                news_event: NewsEvent, market_data: Dict) -> Optional[MarketSignal]:
        """Generate enhanced market signal with sophisticated logic"""
        
        # Extract analysis components
        price_pred = analysis.get('price_predictions', {})
        trading_signals = analysis.get('trading_signals', {})
        impact_assessment = analysis.get('impact_assessment', {})
        
        # Calculate signal strength considering multiple factors
        base_strength = trading_signals.get('signal_strength', 0)
        confidence = analysis.get('confidence_factors', {}).get('overall_confidence', 0.5)
        impact = impact_assessment.get('immediate_impact', 5) / 10.0
        
        # Apply strategy-specific adjustments
        strength = self._apply_strategy_adjustments(base_strength, impact, market_data)
        
        # Determine signal type based on analysis and strategy
        signal_type = self._determine_signal_type(analysis, market_data)
        
        # Calculate duration based on impact and news characteristics
        base_duration = news_event.impact_duration
        duration = int(base_duration * (0.5 + confidence))
        
        # Risk management parameters
        risk_level = trading_signals.get('risk_level', 'medium')
        position_sizing = min(
            trading_signals.get('position_sizing', 0.1),
            self.strategy_params['max_position_size']
        )
        
        # Create signal
        signal = MarketSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            symbol=symbol,
            strength=strength,
            duration=duration,
            confidence=confidence,
            source_agent=self.name,
            risk_level=risk_level,
            expected_return=price_pred.get('magnitude_percent', 0) / 100.0,
            max_position=position_sizing,
            stop_loss=trading_signals.get('stop_loss', self.strategy_params['stop_loss_pct']),
            take_profit=trading_signals.get('take_profit', self.strategy_params['take_profit_pct']),
            sector=self._get_symbol_sector(symbol)
        )
        
        return signal
    
    def _apply_strategy_adjustments(self, base_strength: float, impact: float, 
                                  market_data: Dict) -> float:
        """Apply strategy-specific adjustments to signal strength"""
        
        strength = base_strength
        
        if self.strategy == 'momentum':
            # Boost strength for trending markets
            recent_returns = self._calculate_recent_returns(market_data)
            if recent_returns and abs(recent_returns) > 0.02:
                if (recent_returns > 0 and strength > 0) or (recent_returns < 0 and strength < 0):
                    strength *= 1.3
        
        elif self.strategy == 'mean_reversion':
            # Look for oversold/overbought conditions
            volatility = self._calculate_volatility(market_data)
            if volatility > 0.03:  # High volatility
                strength *= 1.2
        
        elif self.strategy == 'volatility':
            # Amplify strength during uncertain times
            strength *= (1 + impact)
        
        # Apply risk tolerance
        strength *= (0.5 + self.risk_tolerance)
        
        return max(-1.0, min(1.0, strength))
    
    def _determine_signal_type(self, analysis: Dict, market_data: Dict) -> str:
        """Determine optimal signal type based on analysis and market conditions"""
        
        # Get suggested signal type from analysis
        suggested_type = analysis.get('trading_signals', {}).get('signal_type', self.strategy)
        
        # Market condition adjustments
        volatility = self._calculate_volatility(market_data)
        
        if volatility > 0.05:  # High volatility
            return 'volatility'
        elif volatility < 0.01:  # Low volatility
            return 'momentum'
        else:
            return suggested_type
    
    def _validate_signal(self, signal: MarketSignal) -> bool:
        """Validate signal meets agent's criteria"""
        
        # Minimum confidence check
        if signal.confidence < self.strategy_params['min_confidence']:
            return False
        
        # Strength threshold
        if abs(signal.strength) < 0.1:
            return False
        
        # Risk level compatibility
        risk_tolerance_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        if risk_tolerance_map.get(signal.risk_level, 0.5) > self.risk_tolerance + 0.2:
            return False
        
        return True
    
    def _calculate_recent_returns(self, market_data: Dict) -> Optional[float]:
        """Calculate recent returns from market data"""
        # Placeholder - would use actual price history
        current_price = market_data.get('price', 100)
        return random.uniform(-0.05, 0.05)  # Mock recent return
    
    def _calculate_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility"""
        # Placeholder - would use actual price history
        spread = market_data.get('ask', 100) - market_data.get('bid', 99)
        price = market_data.get('price', 100)
        return spread / price if price > 0 else 0.02
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'TSLA': 'Automotive', 'NVDA': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare'
        }
        return sector_map.get(symbol, 'Other')


class PerformanceTracker:
    """Track agent performance and learning"""
    
    def __init__(self):
        self.trades = []
        self.signals = []
        self.performance_metrics = {}
    
    def record_trade(self, trade: Dict):
        """Record trade execution"""
        self.trades.append(trade)
        self._update_metrics()
    
    def record_signal(self, signal: MarketSignal, outcome: Dict = None):
        """Record signal and its outcome"""
        signal_record = {
            'signal': signal.to_dict(),
            'outcome': outcome,
            'timestamp': datetime.now()
        }
        self.signals.append(signal_record)
    
    def _update_metrics(self):
        """Update performance metrics"""
        if not self.trades:
            return
        
        # Calculate basic metrics
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        
        self.performance_metrics = {
            'total_pnl': total_pnl,
            'num_trades': len(self.trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'avg_trade_pnl': total_pnl / len(self.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [trade.get('pnl', 0) for trade in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0


class MarketMemory:
    """Store and retrieve market patterns for learning"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories = []
    
    def store_pattern(self, pattern: Dict):
        """Store market pattern"""
        self.memories.append({
            'pattern': pattern,
            'timestamp': datetime.now()
        })
        
        # Maintain capacity
        if len(self.memories) > self.capacity:
            self.memories = self.memories[-self.capacity:]
    
    def retrieve_similar_patterns(self, current_pattern: Dict, 
                                 similarity_threshold: float = 0.7) -> List[Dict]:
        """Retrieve similar historical patterns"""
        # Simplified similarity matching
        similar = []
        for memory in self.memories:
            if self._calculate_similarity(memory['pattern'], current_pattern) > similarity_threshold:
                similar.append(memory)
        
        return similar[-10:]  # Return most recent similar patterns
    
    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate pattern similarity"""
        # Simplified similarity calculation
        return random.uniform(0.3, 0.9)  # Mock similarity score