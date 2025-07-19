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
import os
from abc import ABC, abstractmethod
import threading
import queue
import time
import sqlite3
from pathlib import Path

# Replace autogen with direct OpenAI API integration
try:
    import openai
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: OpenAI not available. LLM features will be mocked.")

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
    PRODUCT_LAUNCH = "product_launch"


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
    confidence: float  # 0 to 1
    reasoning: str
    risk_level: str  # 'low', 'medium', 'high'
    expected_duration: int  # minutes
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type,
            'symbol': self.symbol,
            'strength': self.strength,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'risk_level': self.risk_level,
            'expected_duration': self.expected_duration,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class LLMInterface:
    """Interface for LLM API calls with fallback to mock responses"""
    
    def __init__(self):
        self.client = None
        if LLM_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your-api-key-here":
                try:
                    self.client = OpenAI(api_key=api_key)
                    logger.info("✅ OpenAI client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
                    self.client = None
            else:
                logger.warning("No valid OpenAI API key found")
        
        if not self.client:
            logger.info("🤖 Using mock LLM responses")
    
    async def generate_response(self, system_prompt: str, user_prompt: str, 
                               model: str = "gpt-4", max_tokens: int = 1000,
                               temperature: float = 0.7) -> str:
        """Generate LLM response with fallback to mock"""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                return self._mock_response(user_prompt)
        else:
            return self._mock_response(user_prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses for testing without API"""
        if "news" in prompt.lower() and "sentiment" in prompt.lower():
            return json.dumps({
                "sentiment_score": random.uniform(-0.5, 0.5),
                "confidence": random.uniform(0.6, 0.9),
                "reasoning": "Mock analysis: Market sentiment appears mixed with moderate uncertainty.",
                "key_factors": ["earnings", "market_conditions", "technical_indicators"]
            })
        elif "trading" in prompt.lower() and "signal" in prompt.lower():
            return json.dumps({
                "action": random.choice(["BUY", "SELL", "HOLD"]),
                "strength": random.uniform(0.3, 0.8),
                "confidence": random.uniform(0.5, 0.9),
                "reasoning": "Mock trading signal based on technical analysis and market conditions.",
                "risk_level": random.choice(["low", "medium", "high"])
            })
        else:
            return "Mock LLM response generated for testing purposes."


class EnhancedLLMNewsAnalyzer:
    """LLM-powered news analyzer using OpenAI API"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.llm = LLMInterface()
        self.analysis_history = []
        
    async def analyze_news(self, news_event: NewsEvent) -> Dict[str, Any]:
        """Analyze news event using LLM"""
        system_prompt = """You are an expert financial analyst specializing in news sentiment analysis and market impact assessment. 
        Your task is to analyze news events and provide structured insights about their potential market impact.
        
        You should consider:
        - The sentiment (positive/negative/neutral) and its strength
        - The potential impact on specific stocks or sectors
        - The likely duration of the impact
        - Risk factors and uncertainty levels
        
        Respond with a JSON object containing:
        - sentiment_score: float between -1 (very negative) and 1 (very positive)
        - confidence: float between 0 and 1
        - reasoning: string explaining your analysis
        - key_factors: list of key factors that influenced your analysis
        - market_impact: string describing expected market impact
        - risk_assessment: string describing potential risks"""
        
        user_prompt = f"""Analyze this news event:
        
        Headline: {news_event.headline}
        Content: {news_event.content}
        Category: {news_event.category.value}
        Affected Symbols: {', '.join(news_event.affected_symbols)}
        Source: {news_event.source}
        
        Please provide a comprehensive analysis of this news event's potential market impact."""
        
        try:
            response = await self.llm.generate_response(system_prompt, user_prompt)
            analysis = json.loads(response)
            
            # Store analysis
            analysis_record = {
                'timestamp': news_event.timestamp,
                'news_id': id(news_event),
                'analysis': analysis
            }
            self.analysis_history.append(analysis_record)
            
            return analysis
            
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            # Fallback analysis
            return {
                'sentiment_score': random.uniform(-0.3, 0.3),
                'confidence': 0.5,
                'reasoning': 'Fallback analysis due to LLM error',
                'key_factors': ['uncertainty'],
                'market_impact': 'Uncertain impact',
                'risk_assessment': 'High uncertainty due to analysis failure'
            }


class AdvancedLLMTradingAgent:
    """Advanced trading agent using LLM for decision making"""
    
    def __init__(self, agent_id: str, strategy_type: str, initial_capital: float,
                 symbols: List[str], risk_tolerance: float = 0.5):
        self.agent_id = agent_id
        self.strategy_type = strategy_type  # 'momentum', 'value', 'arbitrage', etc.
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.symbols = symbols
        self.risk_tolerance = risk_tolerance
        self.llm = LLMInterface()
        self.positions = {symbol: 0 for symbol in symbols}
        self.trade_history = []
        self.performance_metrics = {}
        
    async def generate_trading_signal(self, market_data: Dict, news_analysis: Dict = None) -> MarketSignal:
        """Generate trading signal using LLM analysis"""
        
        system_prompt = f"""You are a professional {self.strategy_type} trader with expertise in quantitative analysis.
        Your risk tolerance is {self.risk_tolerance} (0=very conservative, 1=very aggressive).
        
        You should analyze market data and generate trading signals based on your strategy:
        - {self.strategy_type} strategy principles
        - Current market conditions
        - Risk management considerations
        - Available news and sentiment data
        
        Respond with a JSON object containing:
        - action: "BUY", "SELL", or "HOLD"
        - symbol: the stock symbol to trade
        - strength: float between 0 and 1 (signal strength)
        - confidence: float between 0 and 1
        - reasoning: detailed explanation of your decision
        - risk_level: "low", "medium", or "high"
        - position_size: recommended position size (as fraction of capital)
        - stop_loss: optional stop loss price
        - take_profit: optional take profit price"""
        
        # Prepare market data summary
        market_summary = json.dumps(market_data, indent=2, default=str)
        news_summary = json.dumps(news_analysis, indent=2) if news_analysis else "No recent news"
        
        user_prompt = f"""Current Market Data:
        {market_summary}
        
        Recent News Analysis:
        {news_summary}
        
        Current Positions: {self.positions}
        Available Capital: ${self.current_capital:,.2f}
        
        Based on your {self.strategy_type} strategy and the above information, what trading action do you recommend?"""
        
        try:
            response = await self.llm.generate_response(system_prompt, user_prompt)
            
            # Try to parse JSON, with fallback handling
            try:
                signal_data = json.loads(response)
            except json.JSONDecodeError:
                # If response isn't valid JSON, try to extract structured data
                logger.warning(f"LLM response not valid JSON, using fallback parsing: {response[:100]}...")
                signal_data = self._parse_unstructured_response(response)
            
            # Create MarketSignal object
            signal = MarketSignal(
                timestamp=datetime.now(),
                signal_type=self.strategy_type,
                symbol=signal_data.get('symbol', self.symbols[0]),
                strength=signal_data.get('strength', 0.5),
                confidence=signal_data.get('confidence', 0.5),
                reasoning=signal_data.get('reasoning', 'LLM-generated signal'),
                risk_level=signal_data.get('risk_level', 'medium'),
                expected_duration=30,  # Default 30 minutes
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit')
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            # Fallback signal
            return MarketSignal(
                timestamp=datetime.now(),
                signal_type=self.strategy_type,
                symbol=random.choice(self.symbols),
                strength=random.uniform(0.3, 0.7),
                confidence=0.5,
                reasoning='Fallback signal due to LLM error',
                risk_level='medium',
                expected_duration=30
            )
    
    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """Parse unstructured LLM response and extract trading signal data"""
        signal_data = {
            'symbol': self.symbols[0],
            'strength': 0.5,
            'confidence': 0.5,
            'reasoning': response[:200] + "..." if len(response) > 200 else response,
            'risk_level': 'medium'
        }
        
        # Try to extract some information from text
        if 'buy' in response.lower() or 'bullish' in response.lower():
            signal_data['strength'] = 0.7
        elif 'sell' in response.lower() or 'bearish' in response.lower():
            signal_data['strength'] = 0.3
            
        if 'high confidence' in response.lower():
            signal_data['confidence'] = 0.8
        elif 'low confidence' in response.lower():
            signal_data['confidence'] = 0.3
            
        return signal_data


class RealisticNewsGenerator:
    """Generate realistic market news events"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.news_templates = {
            NewsCategory.EARNINGS: [
                "{company} reports Q{quarter} earnings of ${eps} per share, {beat_miss} estimates",
                "{company} announces {direction} revenue growth in latest quarterly results",
                "Analysts upgrade {company} following earnings beat"
            ],
            NewsCategory.MERGERS: [
                "{company} announces merger agreement with competitor in ${amount}B deal",
                "Regulatory approval pending for {company} merger",
                "{company} exploring strategic alternatives, sources say"
            ],
            NewsCategory.MACRO_ECONOMIC: [
                "Federal Reserve {action} interest rates by {amount} basis points",
                "GDP growth {direction} to {rate}% in latest quarter",
                "Inflation data shows {direction} trend, impacting market sentiment"
            ],
            NewsCategory.PRODUCT_LAUNCH: [
                "{company} unveils revolutionary {product} technology",
                "Innovation in {sector} drives mixed outlook for {company}",
                "{company} announces breakthrough in {technology} development"
            ]
        }
    
    def generate_news_event(self, category: NewsCategory = None, 
                          affected_symbols: List[str] = None) -> NewsEvent:
        """Generate a realistic news event"""
        if category is None:
            category = random.choice(list(NewsCategory))
        
        if affected_symbols is None:
            affected_symbols = random.sample(self.symbols, 
                                           random.randint(1, min(3, len(self.symbols))))
        
        # Generate headline and content based on category
        templates = self.news_templates.get(category, ["Market update affects {company}"])
        template = random.choice(templates)
        
        # Fill in template variables
        company = random.choice(affected_symbols) if affected_symbols else "Company"
        headline = template.format(
            company=company,
            quarter=random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
            eps=f"{random.uniform(0.50, 3.00):.2f}",
            beat_miss=random.choice(['beating', 'missing']),
            direction=random.choice(['strong', 'weak', 'moderate']),
            amount=f"{random.uniform(1.0, 50.0):.1f}",
            action=random.choice(['raises', 'cuts', 'maintains']),
            rate=f"{random.uniform(0.5, 4.0):.1f}",
            product=random.choice(['smartphone', 'laptop', 'AI chip', 'software platform']),
            sector=random.choice(['technology', 'healthcare', 'finance', 'automotive']),
            technology=random.choice(['machine learning', 'quantum computing', 'blockchain'])
        )
        
        # Generate sentiment based on category and keywords
        if any(word in headline.lower() for word in ['beats', 'strong', 'breakthrough', 'revolutionary']):
            sentiment = random.uniform(0.2, 0.8)
        elif any(word in headline.lower() for word in ['misses', 'weak', 'decline', 'regulatory']):
            sentiment = random.uniform(-0.8, -0.2)
        else:
            sentiment = random.uniform(-0.5, 0.5)
        
        return NewsEvent(
            timestamp=datetime.now(),
            category=category,
            headline=headline,
            content=f"Details about {headline.lower()}. Market analysts are monitoring the situation closely.",
            affected_symbols=affected_symbols,
            sentiment_score=sentiment,
            importance=random.uniform(0.3, 1.0),
            confidence=random.uniform(0.6, 0.9)
        )