import csv
import heapq
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import time
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

Priority = str
STATE_PENDING = "PENDING"
STATE_ASSIGNED = "ASSIGNED"
STATE_IN_TRANSIT = "IN_TRANSIT"
STATE_DELIVERED = "DELIVERED"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Order:
    order_id: str
    timestamp: datetime
    location: Tuple[int, int]
    prep_time: int
    priority: Priority
    sla_minutes: int
    state: str = STATE_PENDING
    assigned_agent: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    sla_violation: bool = False

    def estimated_duration(self, travel_time: float) -> float:
        # Add stochastic prep delay: ±10% variation
        variation = random.uniform(0.9, 1.1)
        return travel_time + (self.prep_time * variation)


@dataclass
class Agent:
    agent_id: str
    current_location: Tuple[int, int]
    rating: float
    max_active_orders: int = 2
    active_orders: List[str] = field(default_factory=list)
    cumulative_assignments: int = 0

    @property
    def availability(self) -> bool:
        return len(self.active_orders) < self.max_active_orders

    def add_order(self, order_id: str) -> None:
        self.active_orders.append(order_id)
        self.cumulative_assignments += 1

    def remove_order(self, order_id: str) -> None:
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)


class EnvironmentGraph:
    def __init__(self) -> None:
        self.edges: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = defaultdict(dict)
        self.distance_cache: Optional[Dict[Tuple[int, int], Dict[Tuple[int, int], float]]] = None

    def add_edge(self, source: Tuple[int, int], dest: Tuple[int, int], weight: float) -> None:
        existing = self.edges[source].get(dest)
        if existing is None or weight < existing:
            self.edges[source][dest] = weight

    def add_undirected_edge(self, a: Tuple[int, int], b: Tuple[int, int], weight: float) -> None:
        self.add_edge(a, b, weight)
        self.add_edge(b, a, weight)

    def build_all_pairs(self, threshold: int = 80) -> None:
        nodes = list(self.nodes())
        if len(nodes) > threshold:
            self.distance_cache = None
            return

        dist: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {
            u: {v: float("inf") for v in nodes} for u in nodes
        }
        for u in nodes:
            dist[u][u] = 0.0
            for v, w in self.edges[u].items():
                dist[u][v] = w

        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        self.distance_cache = dist

    def nodes(self) -> set:
        node_set = set(self.edges.keys())
        for neighbors in self.edges.values():
            node_set.update(neighbors.keys())
        return node_set

    def dijkstra(self, source: Tuple[int, int]) -> Dict[Tuple[int, int], float]:
        distances = {node: float("inf") for node in self.nodes()}
        distances[source] = 0.0
        queue: List[Tuple[float, Tuple[int, int]]] = [(0.0, source)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_distance > distances[current_node]:
                continue
            for neighbor, weight in self.edges.get(current_node, {}).items():
                new_distance = current_distance + weight
                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    heapq.heappush(queue, (new_distance, neighbor))
        return distances

    def get_distance(self, source: Tuple[int, int], dest: Tuple[int, int]) -> Optional[float]:
        base_distance = self._get_base_distance(source, dest)
        if base_distance is None:
            return None
        # Simulate network latency: add random delay (1-5 seconds)
        latency = random.uniform(1.0, 5.0)
        return base_distance + latency

    @lru_cache(maxsize=None)
    def _get_base_distance(self, source: Tuple[int, int], dest: Tuple[int, int]) -> Optional[float]:
        if source == dest:
            return 0.0
        if self.distance_cache is not None:
            return self.distance_cache.get(source, {}).get(dest)
        distances = self.dijkstra(source)
        return distances.get(dest)


class PriorityOrderQueue:
    def __init__(self) -> None:
        self.queues: Dict[Priority, deque[Order]] = {
            "high": deque(),
            "normal": deque(),
            "low": deque(),
        }

    def push(self, order: Order) -> None:
        self.queues[order.priority].append(order)

    def pop(self) -> Optional[Order]:
        for priority in PRIORITY_ORDER:
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None

    def __iter__(self):
        for priority in PRIORITY_ORDER:
            yield from self.queues[priority]

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def remove(self, order: Order) -> bool:
        queue = self.queues.get(order.priority)
        if queue is None:
            return False
        try:
            queue.remove(order)
            return True
        except ValueError:
            return False


class MetricsTracker:
    def __init__(self) -> None:
        self.total_orders = 0
        self.completed_orders = 0
        self.delivery_time_sum = 0.0
        self.delivery_time_squared_sum = 0.0
        self.priority_counts = {"high": 0, "normal": 0, "low": 0}
        self.priority_delivery_time = {"high": 0.0, "normal": 0.0, "low": 0.0}
        self.priority_completed = {"high": 0, "normal": 0, "low": 0}
        self.sla_violations = 0
        self.sla_by_priority = {"high": 0, "normal": 0, "low": 0}
        self.queue_warnings: int = 0

    def record_order(self, order: Order) -> None:
        self.total_orders += 1
        self.priority_counts[order.priority] += 1

    def record_delivery(self, order: Order) -> None:
        if order.delivered_at is None:
            return
        duration = (order.delivered_at - order.timestamp).total_seconds() / 60.0
        self.completed_orders += 1
        self.delivery_time_sum += duration
        self.delivery_time_squared_sum += duration * duration
        self.priority_delivery_time[order.priority] += duration
        self.priority_completed[order.priority] += 1
        if order.sla_violation:
            self.sla_violations += 1
            self.sla_by_priority[order.priority] += 1

    def average_delivery_time(self) -> float:
        if self.completed_orders == 0:
            return 0.0
        return self.delivery_time_sum / self.completed_orders

    def delivery_time_stddev(self) -> float:
        if self.completed_orders < 2:
            return 0.0
        mean = self.average_delivery_time()
        variance = (self.delivery_time_squared_sum / self.completed_orders) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def sla_violation_rate(self) -> float:
        if self.completed_orders == 0:
            return 0.0
        return 100.0 * self.sla_violations / self.completed_orders

    def to_dict(self, agents: Dict[str, Agent]) -> Dict:
        assignment_counts = [agent.cumulative_assignments for agent in agents.values()]
        return {
            "summary": {
                "total_orders": self.total_orders,
                "completed_orders": self.completed_orders,
                "avg_delivery_time_minutes": round(self.average_delivery_time(), 2),
                "delivery_time_stddev_minutes": round(self.delivery_time_stddev(), 2),
                "sla_violation_rate_percent": round(self.sla_violation_rate(), 2),
                "queue_warnings": self.queue_warnings,
            },
            "priority_breakdown": {
                priority: {
                    "orders": self.priority_counts[priority],
                    "completed": self.priority_completed[priority],
                    "avg_delivery_time_minutes": round(
                        self.priority_delivery_time[priority] / self.priority_completed[priority], 2
                    ) if self.priority_completed[priority] else 0.0,
                    "sla_violations": self.sla_by_priority[priority],
                }
                for priority in PRIORITY_ORDER
            },
            "fairness": {
                "min_assignments": min(assignment_counts) if assignment_counts else 0,
                "max_assignments": max(assignment_counts) if assignment_counts else 0,
                "assignment_range": (
                    max(assignment_counts) - min(assignment_counts)
                    if assignment_counts else 0
                ),
                "agent_assignments": {
                    agent_id: agent.cumulative_assignments for agent_id, agent in agents.items()
                },
            },
        }


class DispatchSystem:
    def __init__(self, graph: EnvironmentGraph, agents: Dict[str, Agent], orders: List[Order], config: Dict) -> None:
        self.graph = graph
        self.agents = agents
        self.orders = sorted(orders, key=lambda order: order.timestamp)
        self.config = config
        self.metrics = MetricsTracker()
        self.pending_queue = PriorityOrderQueue()
        self.order_by_id = {order.order_id: order for order in self.orders}
        for order in self.orders:
            self.metrics.record_order(order)

        self.graph.build_all_pairs(threshold=self.config.get("all_pairs_threshold", 80))

    def _agent_penalty(self, agent: Agent) -> float:
        return self.config.get("active_order_penalty", 10.0) * len(agent.active_orders)

    def _fairness_bonus(self, agent: Agent) -> float:
        total_assignments = sum(a.cumulative_assignments for a in self.agents.values())
        agent_count = len(self.agents)
        average = total_assignments / agent_count if agent_count else 0.0
        return (average - agent.cumulative_assignments) * self.config.get("fairness_weight", 1.0)

    def _priority_bonus(self, priority: Priority) -> float:
        return self.config.get("priority_weights", {}).get(priority, 0.0)

    def _rating_bonus(self, agent: Agent) -> float:
        return agent.rating * self.config.get("rating_weight", 1.0)

    def generate_candidates(self, current_time: datetime) -> List[Tuple[Agent, Order, float]]:
        candidates: List[Tuple[Agent, Order, float]] = []
        for order in self.pending_orders():
            for agent in self.agents.values():
                if not agent.availability:
                    continue
                travel_time = self.graph.get_distance(agent.current_location, order.location)
                if travel_time is None or travel_time == float("inf"):
                    continue
                estimated_total = travel_time + order.prep_time
                candidates.append((agent, order, estimated_total))
        return candidates

    def pending_orders(self) -> List[Order]:
        return [order for order in self.pending_queue if order.state == STATE_PENDING]

    def score_candidate(self, agent: Agent, order: Order, total_minutes: float) -> float:
        sla_margin = order.sla_minutes - total_minutes
        score = 0.0
        score += self.config.get("delivery_time_weight", -1.0) * total_minutes
        score += self.config.get("sla_weight", 1.0) * sla_margin
        score += self._fairness_bonus(agent)
        score += self._priority_bonus(order.priority)
        score += self._rating_bonus(agent)
        score -= self._agent_penalty(agent)
        return score

    def choose_best_assignment(self, current_time: datetime) -> Optional[Tuple[Agent, Order, float]]:
        if not HAS_SCIPY:
            # Fallback to greedy
            return self._greedy_assignment(current_time)
        return self._optimal_assignment(current_time)

    def _greedy_assignment(self, current_time: datetime) -> Optional[Tuple[Agent, Order, float]]:
        best: Optional[Tuple[Agent, Order, float]] = None
        best_score = float("-inf")
        for order in self.pending_orders():
            for agent in self.agents.values():
                if not agent.availability:
                    continue
                travel_time = self.graph.get_distance(agent.current_location, order.location)
                if travel_time is None or travel_time == float("inf"):
                    continue
                total_minutes = travel_time + order.prep_time
                score = self.score_candidate(agent, order, total_minutes)
                if score > best_score:
                    best_score = score
                    best = (agent, order, total_minutes)
                elif score == best_score and best is not None:
                    if order.priority == "high" and best[1].priority != "high":
                        best = (agent, order, total_minutes)
        return best

    def _optimal_assignment(self, current_time: datetime) -> Optional[Tuple[Agent, Order, float]]:
        pending = self.pending_orders()
        available_agents = [agent for agent in self.agents.values() if agent.availability]
        if not pending or not available_agents:
            return None

        # Build cost matrix: rows=agents, cols=orders, cost=negative score (since Hungarian minimizes)
        cost_matrix = []
        for agent in available_agents:
            row = []
            for order in pending:
                travel_time = self.graph.get_distance(agent.current_location, order.location)
                if travel_time is None or travel_time == float("inf"):
                    row.append(float("inf"))  # Unfeasible
                else:
                    total_minutes = travel_time + order.prep_time
                    score = self.score_candidate(agent, order, total_minutes)
                    row.append(-score)  # Minimize negative score = maximize score
            cost_matrix.append(row)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i][j] != float("inf"):
                agent = available_agents[i]
                order = pending[j]
                travel_time = self.graph.get_distance(agent.current_location, order.location)
                total_minutes = travel_time + order.prep_time
                return (agent, order, total_minutes)
        return None

    def assign_order(self, agent: Agent, order: Order, current_time: datetime, travel_minutes: float) -> datetime:
        order.state = STATE_ASSIGNED
        order.assigned_agent = agent.agent_id
        order.assigned_at = current_time
        actual_prep = order.estimated_duration(0)  # Get stochastic prep
        order.completion_time = current_time + timedelta(minutes=travel_minutes + actual_prep)
        agent.add_order(order.order_id)
        return order.completion_time

    def complete_order(self, agent: Agent, order: Order, completion_time: datetime) -> None:
        order.state = STATE_DELIVERED
        order.delivered_at = completion_time
        order.sla_violation = (completion_time - order.timestamp).total_seconds() / 60.0 > order.sla_minutes
        agent.remove_order(order.order_id)
        agent.current_location = order.location
        self.metrics.record_delivery(order)

    def process_pending_assignments(self, current_time: datetime, events: List[Tuple[datetime, str, str]]) -> None:
        while True:
            assignment = self.choose_best_assignment(current_time)
            if assignment is None:
                break
            agent, order, total_minutes = assignment
            self.pending_queue.remove(order)
            completion_time = self.assign_order(agent, order, current_time, total_minutes)
            heapq.heappush(events, (completion_time, agent.agent_id, order.order_id))

    def run(self) -> Dict:
        events: List[Tuple[datetime, str, str]] = []
        order_index = 0
        current_time = self.orders[0].timestamp if self.orders else datetime.now()

    def run(self) -> Dict:
        events: List[Tuple[datetime, str, str]] = []
        order_index = 0
        current_time = self.orders[0].timestamp if self.orders else datetime.now()

        while order_index < len(self.orders) or events:
            next_order_time = self.orders[order_index].timestamp if order_index < len(self.orders) else None
            next_event_time = events[0][0] if events else None

            if next_event_time is not None and (next_order_time is None or next_event_time <= next_order_time):
                current_time, agent_id, order_id = heapq.heappop(events)
                agent = self.agents[agent_id]
                order = self.order_by_id[order_id]
                self.complete_order(agent, order, current_time)
                self.process_pending_assignments(current_time, events)
            elif order_index < len(self.orders):
                order = self.orders[order_index]
                order_index += 1
                current_time = order.timestamp
                self.pending_queue.push(order)
                self.process_pending_assignments(current_time, events)
                if len(self.pending_queue) > 0 and all(not agent.availability for agent in self.agents.values()):
                    self.metrics.queue_warnings += 1

        return {
            "metrics": self.metrics.to_dict(self.agents),
            "orders": [self.order_to_dict(order) for order in self.orders],
        }

        return {
            "metrics": self.metrics.to_dict(self.agents),
            "orders": [self.order_to_dict(order) for order in self.orders],
        }

    @staticmethod
    def order_to_dict(order: Order) -> Dict:
        return {
            "order_id": order.order_id,
            "priority": order.priority,
            "state": order.state,
            "assigned_agent": order.assigned_agent,
            "assigned_at": order.assigned_at.isoformat() if order.assigned_at else None,
            "completed_at": order.completion_time.isoformat() if order.completion_time else None,
            "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
            "sla_violation": order.sla_violation,
        }


def parse_orders(path: str) -> List[Order]:
    orders: List[Order] = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                orders.append(Order(
                    order_id=row["order_id"].strip(),
                    timestamp=datetime.fromisoformat(row["timestamp"].strip()),
                    location=(int(row["location_x"]), int(row["location_y"])),
                    prep_time=int(row["prep_time_minutes"]),
                    priority=row["priority"].strip().lower(),
                    sla_minutes=int(row["sla_minutes"]),
                ))
            except Exception as exc:
                print(f"Skipping invalid order row: {row} ({exc})")
    return orders


def parse_agents(path: str) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                agent_id = row["agent_id"].strip()
                agents[agent_id] = Agent(
                    agent_id=agent_id,
                    current_location=(int(row["current_x"]), int(row["current_y"])),
                    rating=float(row["rating"]),
                )
            except Exception as exc:
                print(f"Skipping invalid agent row: {row} ({exc})")
    return agents


def parse_environment(path: str) -> EnvironmentGraph:
    graph = EnvironmentGraph()
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                source = (int(row["from_x"]), int(row["from_y"]))
                dest = (int(row["to_x"]), int(row["to_y"]))
                distance = float(row["distance_minutes"])
                multiplier = float(row.get("delay_multiplier", 1.0))
                travel_time = distance * multiplier
                graph.add_undirected_edge(source, dest, travel_time)
            except Exception as exc:
                print(f"Skipping invalid environment row: {row} ({exc})")
    return graph


def parse_constraints(path: str) -> Dict[str, float]:
    constraints: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row.get("constraint", "").strip()
            raw_value = row.get("value", "").strip()
            if not key:
                continue
            try:
                if raw_value.isdigit():
                    value = int(raw_value)
                else:
                    value = float(raw_value)
            except ValueError:
                value = raw_value
            constraints[key] = value
    return constraints


def apply_constraints_to_config(config: Dict, constraints: Dict[str, float]) -> Dict:
    for key, value in constraints.items():
        if key.startswith("priority_weight_"):
            priority = key.rsplit("_", 1)[-1]
            config.setdefault("priority_weights", {})[priority] = float(value)
        else:
            config[key] = value
    return config


def parse_orders(path: str, default_sla_minutes: int = 50) -> List[Order]:
    orders: List[Order] = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                sla_raw = row.get("sla_minutes", "").strip()
                sla_minutes = int(sla_raw) if sla_raw else default_sla_minutes
                orders.append(Order(
                    order_id=row["order_id"].strip(),
                    timestamp=datetime.fromisoformat(row["timestamp"].strip()),
                    location=(int(row["location_x"]), int(row["location_y"])),
                    prep_time=int(row["prep_time_minutes"]),
                    priority=row["priority"].strip().lower(),
                    sla_minutes=sla_minutes,
                ))
            except Exception as exc:
                print(f"Skipping invalid order row: {row} ({exc})")
    return orders


def parse_agents(path: str, max_active_orders: int = 2) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                agent_id = row["agent_id"].strip()
                agents[agent_id] = Agent(
                    agent_id=agent_id,
                    current_location=(int(row["current_x"]), int(row["current_y"])),
                    rating=float(row["rating"]),
                    max_active_orders=max_active_orders,
                )
            except Exception as exc:
                print(f"Skipping invalid agent row: {row} ({exc})")
    return agents


def default_config() -> Dict:
    return {
        "all_pairs_threshold": 80,
        "delivery_time_weight": -1.0,
        "sla_weight": 1.4,
        "fairness_weight": 0.8,
        "priority_weights": {"high": 40.0, "normal": 15.0, "low": 0.0},
        "rating_weight": 2.0,
        "active_order_penalty": 8.0,
        "max_active_orders_per_agent": 2,
        "default_sla_minutes": 50,
        "decision_latency_target_seconds": 5,
    }


def write_metrics(output_path: str, results: Dict) -> None:
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
