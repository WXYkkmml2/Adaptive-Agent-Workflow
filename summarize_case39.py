"""Summarize complete or partial case39 CSV batches without mixing protocols."""
import argparse
import csv
import json
import math
from collections import defaultdict

METRICS = ("real_actions_used", "wasted_actions", "llm_calls", "total_tokens",
           "catalog_tokens", "illegal_tool_calls", "zone3_touched")


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["formal"], row["model"], row["base_url"],
                row["max_actions"], row["protocol_hash"])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        for subset, items in (("all", group), ("hidden_limit_triggered", [r for r in group if int(r["hidden_limit_triggered"])])):
            n = len(items)
            wins = sum(int(r["success"]) for r in items)
            output.append({"method": key[0], "formal": int(key[1]), "model": key[2], "base_url": key[3],
                           "max_actions": int(key[4]), "protocol_hash": key[5], "subset": subset, "n": n,
                           "success_rate": wins/n if n else None, "wilson_95": wilson(wins, n),
                           **{metric: sum(float(r[metric]) for r in items)/n if n else None for metric in METRICS}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file")
    args = parser.parse_args()
    with open(args.csv_file, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["method"], r["repeat"]) for r in rows]
    if len(set(keys)) != len(keys):
        parser.error("Duplicate (method, repeat) rows")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
