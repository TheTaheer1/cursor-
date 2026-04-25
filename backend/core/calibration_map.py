"""
calibration_map.py
Topic graph where each node is a (topic, question_type, difficulty_tier) triple.
Nodes are coloured by zone:
  - Zone C: high confidence + wrong (most dangerous)
  - Zone B: uncertain + wrong
  - Green:  correct
The adversary reads this map to target weak nodes; the frontend visualises it.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional


ZONE_C = "zone_c"
ZONE_B = "zone_b"
ZONE_GREEN = "green"


class CalibrationNode:
    """A single node in the calibration map representing a (topic, type, tier) triple."""

    def __init__(self, topic: str, question_type: str, difficulty_tier: str):
        self.topic: str = topic
        self.question_type: str = question_type
        self.difficulty_tier: str = difficulty_tier
        self.zone: str = ZONE_B
        self.confidence_history: List[float] = []
        self.correctness_history: List[bool] = []
        self.visit_count: int = 0
        self.last_updated: float = time.time()
        self.previous_zone: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.topic}::{self.question_type}::{self.difficulty_tier}"

    @property
    def confidence_avg(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def accuracy_avg(self) -> float:
        if not self.correctness_history:
            return 0.0
        return sum(1 for c in self.correctness_history if c) / len(self.correctness_history)

    def update(self, zone: str, confidence: float, is_correct: bool) -> None:
        self.previous_zone = self.zone
        self.zone = zone
        self.confidence_history.append(float(confidence))
        self.correctness_history.append(bool(is_correct))
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        if len(self.correctness_history) > 10:
            self.correctness_history.pop(0)
        self.visit_count += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "topic": self.topic,
            "question_type": self.question_type,
            "difficulty_tier": self.difficulty_tier,
            "zone": self.zone,
            "previous_zone": self.previous_zone,
            "confidence_avg": round(self.confidence_avg, 3),
            "accuracy_avg": round(self.accuracy_avg, 3),
            "visit_count": self.visit_count,
            "last_updated": self.last_updated,
        }


class CalibrationMap:
    """Manages all calibration nodes and exposes queries for the adversary and UI."""

    def __init__(self):
        self.nodes: Dict[str, CalibrationNode] = {}
        self._seed_starter_nodes()

    def _seed_starter_nodes(self) -> None:
        starters = [
            ("math", "math", "moderate"),
            ("math", "math", "hard"),
            ("math", "math", "expert"),
            ("math", "reasoning", "moderate"),
            ("code", "code", "moderate"),
            ("code", "code", "hard"),
            ("code", "code", "expert"),
            ("code", "reasoning", "hard"),
            ("logic", "reasoning", "moderate"),
            ("logic", "reasoning", "hard"),
            ("logic", "reasoning", "expert"),
            ("factual", "factual", "moderate"),
            ("factual", "factual", "hard"),
            ("factual", "factual", "expert"),
            ("factual", "reasoning", "moderate"),
            ("planning", "reasoning", "moderate"),
            ("planning", "reasoning", "hard"),
            ("planning", "reasoning", "expert"),
            ("planning", "factual", "moderate"),
            ("planning", "code", "hard"),
        ]
        for topic, qtype, tier in starters:
            node = CalibrationNode(topic, qtype, tier)
            self.nodes[node.key] = node

    def get_or_create(self, topic: str, question_type: str, difficulty_tier: str) -> CalibrationNode:
        key = f"{topic}::{question_type}::{difficulty_tier}"
        if key not in self.nodes:
            self.nodes[key] = CalibrationNode(topic, question_type, difficulty_tier)
        return self.nodes[key]

    def update_node(
        self,
        topic: str,
        question_type: str,
        difficulty_tier: str,
        zone: str,
        confidence: float,
        is_correct: bool,
    ) -> CalibrationNode:
        node = self.get_or_create(topic, question_type, difficulty_tier)
        node.update(zone, confidence, is_correct)
        return node

    def get_zone_c_nodes(self) -> List[CalibrationNode]:
        nodes = [n for n in self.nodes.values() if n.zone == ZONE_C]
        return sorted(nodes, key=lambda n: n.confidence_avg, reverse=True)

    def get_zone_b_nodes(self) -> List[CalibrationNode]:
        nodes = [n for n in self.nodes.values() if n.zone == ZONE_B]
        return sorted(nodes, key=lambda n: n.visit_count)

    def get_green_nodes(self) -> List[CalibrationNode]:
        return [n for n in self.nodes.values() if n.zone == ZONE_GREEN]

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes.values()]}

    def snapshot(self) -> dict:
        zone_c = sum(1 for n in self.nodes.values() if n.zone == ZONE_C)
        zone_b = sum(1 for n in self.nodes.values() if n.zone == ZONE_B)
        green = sum(1 for n in self.nodes.values() if n.zone == ZONE_GREEN)
        return {
            "timestamp": time.time(),
            "zone_c_count": zone_c,
            "zone_b_count": zone_b,
            "green_count": green,
            "total_nodes": len(self.nodes),
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }
