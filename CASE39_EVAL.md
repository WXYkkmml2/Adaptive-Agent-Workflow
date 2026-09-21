# case39 regional voltage restoration evaluation

Run with a configured `LLM_API_KEY` and the same `LLM_MODEL`/`LLM_BASE_URL` for all methods:

```sh
.venv/bin/python run_case39.py --repeats 10 --output case39_results.csv
```

The script validates the fixture offline before any model call. The witness is never included in a prompt or tool catalog. The thresholds and temperature are fixed in `config/settings.py`. Each repeat runs `hierarchical`, `two_layer`, and `two_layer_full_restart` on fresh copies of the same case39 fault. All three use the same candidate validator, sequential joint simulation gate, real action interface, and global goal check. Only the hierarchical method narrows the catalog to the local zone and retains completed region actions after a deviation. The full restart baseline resets the physical state but keeps the cumulative action log and six-action budget.

CSV definitions: `success` requires all voltage and line constraints, at most six requested real actions, and no region 3 operation. `real_actions_used` counts requested operations including clipped/failed ones. `wasted_actions` counts unique requests discarded by a full restart, clipped, ineffective, or causing a worse violation. `replanned_tasks` records the region or whole plan selected for retry. `catalog_tokens` is a deterministic UTF-8 JSON length/4 estimate; `total_tokens` uses the API response usage. `proposed_actions` preserves every returned candidate.

The legacy `run_compare.py` measures a case14 smoke task and is not evidence for this evaluation.
