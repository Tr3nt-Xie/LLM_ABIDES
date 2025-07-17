#!/usr/bin/env python3
"""
Mock AutoGen Module
==================

Mock implementation of autogen functionality for ABIDES-LLM integration.
This provides the minimal interface needed when the real autogen package is not available.
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class MockAgent:
    """Mock AutoGen agent for testing purposes"""
    
    def __init__(self, name: str, system_message: str = "", **kwargs):
        self.name = name
        self.system_message = system_message
        self.conversation_history = []
        
    def generate_reply(self, message: str, sender: Optional['MockAgent'] = None) -> str:
        """Generate a mock reply"""
        # Simple mock responses based on message content
        message_lower = message.lower()
        
        if "analyze" in message_lower and "market" in message_lower:
            return self._generate_market_analysis()
        elif "news" in message_lower:
            return self._generate_news_analysis()
        elif "trading" in message_lower or "buy" in message_lower or "sell" in message_lower:
            return self._generate_trading_decision()
        else:
            return f"Mock response from {self.name}: I understand your request about '{message[:50]}...'"
    
    def _generate_market_analysis(self) -> str:
        """Generate mock market analysis"""
        analyses = [
            "The market shows moderate volatility with mixed signals. Technical indicators suggest a neutral trend.",
            "Current market conditions indicate increased uncertainty. Volume patterns suggest institutional activity.",
            "Price action demonstrates typical intraday fluctuations. Support levels remain intact.",
            "Market sentiment appears cautiously optimistic with some defensive positioning evident."
        ]
        return random.choice(analyses)
    
    def _generate_news_analysis(self) -> str:
        """Generate mock news analysis"""
        analyses = [
            "The news appears to have mixed market implications with moderate impact expected.",
            "This development could influence sector sentiment but broader market impact remains unclear.",
            "News sentiment suggests neutral to slightly positive market reaction in the near term.",
            "The announcement contains both positive and negative elements requiring careful analysis."
        ]
        return random.choice(analyses)
    
    def _generate_trading_decision(self) -> str:
        """Generate mock trading decision"""
        decisions = [
            "Recommend maintaining current position with close monitoring of price levels.",
            "Suggest taking a modest long position with tight risk management parameters.",
            "Consider reducing exposure given current market uncertainty and volatility.",
            "Hold strategy appears optimal given mixed signals and unclear trend direction."
        ]
        return random.choice(decisions)


class MockGroupChat:
    """Mock GroupChat for multi-agent conversations"""
    
    def __init__(self, agents: List[MockAgent], messages: List[Dict] = None):
        self.agents = agents
        self.messages = messages or []
        
    def add_message(self, message: str, sender: str):
        """Add a message to the group chat"""
        self.messages.append({
            "sender": sender,
            "message": message,
            "timestamp": "mock_timestamp"
        })


class MockGroupChatManager:
    """Mock GroupChatManager for orchestrating conversations"""
    
    def __init__(self, groupchat: MockGroupChat, llm_config: Dict = None):
        self.groupchat = groupchat
        self.llm_config = llm_config or {}
        
    def initiate_chat(self, message: str, **kwargs) -> List[Dict]:
        """Initiate a mock chat conversation"""
        responses = []
        
        # Simulate each agent responding
        for agent in self.groupchat.agents:
            response = agent.generate_reply(message)
            responses.append({
                "agent": agent.name,
                "response": response
            })
            self.groupchat.add_message(response, agent.name)
        
        return responses


# Mock classes for compatibility
class ConversableAgent(MockAgent):
    """Mock ConversableAgent"""
    pass


class AssistantAgent(MockAgent):
    """Mock AssistantAgent"""
    pass


class UserProxyAgent(MockAgent):
    """Mock UserProxyAgent"""
    pass


# Configuration helpers
def config_list_from_json(json_file: str = None, **kwargs) -> List[Dict]:
    """Mock configuration loader"""
    return [{
        "model": "mock-gpt-4",
        "api_key": "mock-api-key",
        "api_type": "mock"
    }]


def filter_config(config_list: List[Dict], **kwargs) -> List[Dict]:
    """Mock config filter"""
    return config_list


# Export the mock classes and functions
__all__ = [
    'MockAgent',
    'MockGroupChat', 
    'MockGroupChatManager',
    'ConversableAgent',
    'AssistantAgent', 
    'UserProxyAgent',
    'config_list_from_json',
    'filter_config'
]


# Aliases for compatibility
Agent = MockAgent
GroupChat = MockGroupChat  
GroupChatManager = MockGroupChatManager