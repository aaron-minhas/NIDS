"""Scenario presets for the AIES NIDS Mixed Attack Lab."""
from __future__ import annotations


SCENARIO_PROFILES = {
    "Executive mixed": {
        "description": "Balanced board-room demo: benign baseline plus every attack family available.",
        "weights": {
            "BENIGN": 2,
            "DDoS": 2,
            "DoS": 2,
            "PortScan": 2,
            "BruteForce": 2,
            "Botnet": 1,
            "WebAttack": 1,
            "Infiltration": 1,
        },
    },
    "Availability storm": {
        "description": "Service availability incident with DDoS/DoS pressure and supporting recon.",
        "weights": {"BENIGN": 1, "DDoS": 5, "DoS": 3, "PortScan": 1},
    },
    "Credential pressure": {
        "description": "Identity/SOC story focused on brute force and suspicious host behaviour.",
        "weights": {"BENIGN": 2, "BruteForce": 5, "Botnet": 2, "PortScan": 1},
    },
    "Web perimeter": {
        "description": "AppSec story for public services: web attacks with scans and background traffic.",
        "weights": {"BENIGN": 2, "WebAttack": 5, "PortScan": 2, "BruteForce": 1},
    },
    "Recon to infiltration": {
        "description": "Investigation story showing scan, attempted access, and rare infiltration evidence.",
        "weights": {"BENIGN": 2, "PortScan": 4, "BruteForce": 2, "Infiltration": 2},
    },
    "Benign baseline": {
        "description": "Control run for explaining normal traffic and why rules should stay quiet.",
        "weights": {"BENIGN": 1},
    },
}


def scenario_names() -> list[str]:
    return list(SCENARIO_PROFILES.keys())


def scenario_description(scenario: str) -> str:
    profile = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["Executive mixed"])
    return str(profile["description"])


def scenario_targets(max_rows: int, scenario: str) -> dict[str, int]:
    profile = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES["Executive mixed"])
    weights = profile["weights"]
    total_weight = max(1, sum(int(v) for v in weights.values()))
    targets: dict[str, int] = {}
    assigned = 0
    items = list(weights.items())

    for label, weight in items:
        count = max(1, int(max_rows * int(weight) / total_weight))
        targets[label] = count
        assigned += count

    while assigned < max_rows and items:
        for label, _ in items:
            if assigned >= max_rows:
                break
            targets[label] += 1
            assigned += 1

    return targets
