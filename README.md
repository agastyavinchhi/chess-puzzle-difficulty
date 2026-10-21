# Chess Puzzle Difficulty

Predicts a [Lichess puzzle](https://database.lichess.org/#puzzles)'s rating from its board position and solution alone, using a gradient-boosted regression model (LightGBM).

**Result:** MAE of **279 rating points** (R² = 0.43) on 66,127 held-out puzzles, 27% lower than always predicting the mean, using 15 hand-built features and no chess engine.

## Data

- **Source:** the full Lichess puzzle database (~6.16M puzzles, ~300 MB compressed). `prepare.py` streams it and randomly samples 500,000 puzzles (seed 42).
- **Cleaning:** keeps only puzzles with `RatingDeviation ≤ 90` and `NbPlays ≥ 100`, so the ratings used as labels are reliable. 330,631 remain, split 80/20 into train and test.
- **Target:** `Rating`, which ranges from about 400 to 3,000 (mean ~1,487, std ~464).
- **No leakage:** columns that come from people solving the puzzle (`Popularity`, `NbPlays`, `RatingDeviation`, `Themes`, `DailyDate`) are never used as features.

## Features

All 15 features are computed with `python-chess` from the position the solver faces.

| Group | Features |
|---|---|
| Solution | `n_solver_moves`, `frac_forcing` (share of checks and captures) |
| Key move | `key_is_capture`, `key_is_check`, `key_is_quiet`, `key_is_promotion`, `key_is_sacrifice`, `key_piece_type` |
| Board | `total_material`, `material_balance`, `n_legal_moves`, `n_attacked_pieces`, `white_to_move` |
| Context | `game_phase`, `has_opening_tag` |

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (mean) | 462.9 | 383.4 | 0.00 |
| Ridge | 368.7 | 295.8 | 0.37 |
| **LightGBM** | **348.2** | **279.1** | **0.43** |

LightGBM was tuned with a 12-configuration grid search over `num_leaves`, `learning_rate` and `min_child_samples`, using early stopping on a validation slice. The best setup was 63 leaves, a learning rate of 0.03 and 902 trees.

**Key findings**

- **More data barely helped.** Going from a 50K to a 500K puzzle sample (10× more training data) improved MAE by only ~5 points (284.5 → 279.1). The model is limited by its features, not by the amount of data.
- **Solution shape matters most.** The top features by mean |SHAP| are `frac_forcing` (172; more checks and captures make a puzzle easier) and `n_solver_moves` (119; longer solutions make it harder).
- **Predictions regress toward the mean.** The model is most accurate between 1000 and 2000, but overrates easy puzzles and underrates very hard ones (MAE 712 above 2500). Shallow board statistics can't capture what makes elite puzzles hard.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

On first launch the app runs `prepare.py` and `train.py` to download the data and build the model. The dashboard shows the metrics, predicted vs. actual ratings, error by rating band, feature importance, and a puzzle explorer.

| File | Purpose |
|---|---|
| `prepare.py` | Download, sample, clean, and compute features → `data/features.parquet` |
| `train.py` | Train the models and save metrics, predictions, SHAP importance and the model → `artifacts/` |
| `app.py` | Streamlit dashboard |
