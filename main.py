import argparse
import json
import os
from pathlib import Path

from dispatch import (
    Agent,
    DispatchSystem,
    Order,
    apply_constraints_to_config,
    default_config,
    parse_agents,
    parse_constraints,
    parse_environment,
    parse_orders,
    write_metrics,
)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return default_config()
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    default = default_config()
    default.update(config)
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Delivery Dispatch Simulator")
    parser.add_argument("--orders", default="data/raw/orders.csv", help="Path to orders CSV")
    parser.add_argument("--agents", default="data/raw/agents.csv", help="Path to agents CSV")
    parser.add_argument("--environment", default="data/raw/environment_edges.csv", help="Path to environment CSV")
    parser.add_argument("--constraints", default="data/raw/constraints.csv", help="Path to constraints CSV")
    parser.add_argument("--config", default="config.json", help="Optional config JSON file")
    parser.add_argument("--output", default="dispatch_metrics.json", help="Output metrics JSON file")
    parser.add_argument("--runs", type=int, default=10, help="Number of Monte Carlo runs")
    args = parser.parse_args()

    config = load_config(args.config)
    if os.path.exists(args.constraints):
        constraints = parse_constraints(args.constraints)
        config = apply_constraints_to_config(config, constraints)

    orders = parse_orders(args.orders, default_sla_minutes=int(config.get("default_sla_minutes", 50)))
    agents = parse_agents(args.agents, max_active_orders=int(config.get("max_active_orders_per_agent", 2)))
    graph = parse_environment(args.environment)

    # Monte Carlo: Run multiple simulations and average metrics
    all_results = []
    for run in range(args.runs):
        print(f"Running simulation {run + 1}/{args.runs}")
        # Reset agents and orders for each run
        run_agents = {aid: Agent(a.agent_id, a.current_location, a.rating) for aid, a in agents.items()}
        run_orders = [Order(o.order_id, o.timestamp, o.location, o.prep_time, o.priority, o.sla_minutes) for o in orders]
        pipeline = DispatchSystem(graph, run_agents, run_orders, config)
        result = pipeline.run()
        all_results.append(result)

    # Average metrics
    averaged_metrics = average_results(all_results)
def average_results(results_list: list) -> dict:
    if not results_list:
        return {}
    num_runs = len(results_list)
    averaged = {"metrics": {}, "orders": results_list[0]["orders"]}  # Orders are the same, take first
    metrics_keys = results_list[0]["metrics"].keys()
    for key in metrics_keys:
        if key == "priority_breakdown":
            averaged["metrics"][key] = {}
            for prio in ["high", "normal", "low"]:
                averaged["metrics"][key][prio] = {}
                for subkey in results_list[0]["metrics"][key][prio].keys():
                    values = [r["metrics"][key][prio][subkey] for r in results_list]
                    averaged["metrics"][key][prio][subkey] = sum(values) / num_runs
        elif key == "fairness":
            averaged["metrics"][key] = {}
            for subkey in results_list[0]["metrics"][key].keys():
                values = [r["metrics"][key][subkey] for r in results_list]
                averaged["metrics"][key][subkey] = sum(values) / num_runs
        else:
            values = [r["metrics"][key] for r in results_list]
            averaged["metrics"][key] = sum(values) / num_runs
    return averaged
    print(f"Simulation complete. Metrics written to {args.output}")


if __name__ == "__main__":
    main()
