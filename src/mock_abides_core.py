#!/usr/bin/env python3
"""
Mock ABIDES Core
================

Lightweight stand-in for ABIDES Core agent interfaces to allow development
without installing the full ABIDES stack. This mirrors key workflows:
- TradingAgent base class with kernel lifecycle hooks
- Message class for agent messaging
- util.log_print helper

Note: This is NOT a full kernel. sendMessage stores messages in outbox and,
if a recipient is registered in the registry, delivers immediately by calling
its receiveMessage for convenience in single-process tests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class util:
	@staticmethod
	def log_print(msg: str) -> None:
		print(msg)


class Message:
	def __init__(self, msg_type: str, body: Optional[Dict[str, Any]] = None):
		self.msg_type = msg_type
		self.body = body or {}


class _AgentRegistry:
	"""Simple in-memory registry to emulate kernel routing in tests."""
	_registry: Dict[Any, Any] = {}

	@classmethod
	def register(cls, agent_id: Any, agent_obj: Any) -> None:
		cls._registry[agent_id] = agent_obj

	@classmethod
	def get(cls, agent_id: Any) -> Optional[Any]:
		return cls._registry.get(agent_id)


class TradingAgent:
	"""Minimal TradingAgent compatible interface."""
	
	def __init__(self, id, name, type, random_state=None, log_orders: bool = False):
		self.id = id
		self.name = name
		self.type = type
		self.random_state = random_state
		self.log_orders = log_orders
		self._wake_frequency = None
		self._inbox = []
		self._outbox = []
		_AgentRegistry.register(id, self)
	
	# Kernel lifecycle hooks
	def kernelStarting(self, startTime: datetime) -> None:
		pass
	
	def kernelStopping(self) -> None:
		pass
	
	# Messaging
	def receiveMessage(self, currentTime: datetime, msg: Message) -> None:
		self._inbox.append((currentTime, msg))
	
	def sendMessage(self, recipient_id, msg: Message, delay: float = 0.0, broadcast: bool = False) -> None:
		self._outbox.append((recipient_id, msg))
		if broadcast:
			# naive broadcast to all registered agents except self
			for rid, agent in list(_AgentRegistry._registry.items()):
				if rid == self.id:
					continue
				try:
					agent.receiveMessage(datetime.utcnow(), msg)
				except Exception:
					logger.debug("Broadcast delivery failed", exc_info=True)
			return
		# direct delivery if recipient registered
		rec = _AgentRegistry.get(recipient_id)
		if rec is not None:
			try:
				rec.receiveMessage(datetime.utcnow(), msg)
			except Exception:
				logger.debug("Direct delivery failed", exc_info=True)
	
	# Wake/scheduling
	def wakeup(self, currentTime: datetime) -> None:
		pass
	
	def getWakeFrequency(self):
		return self._wake_frequency
	
	def setWakeFrequency(self, freq):
		self._wake_frequency = freq