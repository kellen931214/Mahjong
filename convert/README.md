# MJAI Converter

This converter preserves the existing 1380-dimensional state features and
181-action encoding while generating the unified reward and discounted RTG.

Install dependencies:

```bash
python3 -m pip install -r convert/requirements.txt
```

Run a small validation conversion into a new output directory:

```bash
python3 convert/convert_mjson_to_features.py \
  --data-dir /workspace/Mahjong/data/mjai/2024 \
  --output-dir /data/converted_features_reward_v3_smoke \
  --max-files 10
```

Then run the full conversion:

```bash
python3 convert/convert_mjson_to_features.py \
  --data-dir /workspace/Mahjong/data/mjai/2024 \
  --output-dir /data/converted_features_reward_v3
```

Each chunk contains:

- `features.npy`
- `actions.npy`
- `rtgs.npy`
- `trajectory_boundaries.npy`
- `rewards_debug.npy`

Always use a new output directory. The converter refuses to mix newly
generated chunks with an older reward/RTG dataset.
