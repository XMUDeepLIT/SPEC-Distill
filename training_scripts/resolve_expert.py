#!/usr/bin/env python3
"""Resolve an expert data path from expert_registry.yaml by ID.

Usage:
    python3 resolve_expert.py <expert_id> [--field <field_name>]

Examples:
    python3 resolve_expert.py 14b-think-16k             # prints the path
    python3 resolve_expert.py 14b-think-16k --field path # same as above
    python3 resolve_expert.py 14b-think-16k --field teacher_model
    python3 resolve_expert.py --list                     # list all expert IDs
"""
import sys
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(SCRIPT_DIR, "expert_registry.yaml")


def load_registry():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)["experts"]


def main():
    if "--list" in sys.argv:
        experts = load_registry()
        for eid, info in experts.items():
            teacher = info.get("teacher_model", "?")
            think = info.get("think_mode", "?")
            resp = info.get("max_response_length", "?")
            samples = info.get("num_samples", "?")
            print(f"  {eid:<25s}  {teacher:<18s}  {think:<10s}  resp={resp}  samples={samples}")
        return

    if len(sys.argv) < 2:
        print("Usage: python3 resolve_expert.py <expert_id> [--field <field>]", file=sys.stderr)
        sys.exit(1)

    expert_id = sys.argv[1]
    field = "path"
    if "--field" in sys.argv:
        field = sys.argv[sys.argv.index("--field") + 1]

    experts = load_registry()
    if expert_id not in experts:
        print(f"ERROR: Expert '{expert_id}' not found in {REGISTRY}", file=sys.stderr)
        print(f"Available experts:", file=sys.stderr)
        for eid in experts:
            print(f"  - {eid}", file=sys.stderr)
        sys.exit(1)

    value = experts[expert_id].get(field)
    if value is None:
        print(f"ERROR: Field '{field}' not found for expert '{expert_id}'", file=sys.stderr)
        sys.exit(1)

    if isinstance(value, str):
        value = os.path.expandvars(os.path.expanduser(value))

    print(value)


if __name__ == "__main__":
    main()
