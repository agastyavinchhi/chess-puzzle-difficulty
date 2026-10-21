# Chess Puzzle Difficulty

How hard is a chess puzzle, judged only by looking at it? This project trains a gradient-boosted regression model to predict a [Lichess puzzle](https://database.lichess.org/#puzzles)'s Glicko-2 `Rating` from the position and its solution line. The model never sees anything that comes from people solving the puzzle (popularity, play counts, rating deviation, themes, or the daily date), so it has to judge difficulty from chess structure alone.

**Result:** LightGBM predicts puzzle rating with an MAE of **279 points** (R² = 0.43) on 66,127 held-out puzzles. That is 27% better than always guessing the mean, using 15 hand-built features and no chess engine.

## The problem

- **Task:** supervised regression. Each puzzle has a rating between about 400 and 2,900. In this sample the mean is about 1,485 and the standard deviation is about 465.
- **Input:** the FEN (the position before the opponent's setup move) and the UCI move list. The first move is the opponent's. After it, the solver plays every other move.
- **Target:** `Rating`, which Lichess estimates from thousands of solve attempts.
- **Leakage guard:** `Popularity`, `NbPlays`, `RatingDeviation`, `Themes` and `DailyDate` are all derived from solver behaviour or human tagging, so none of them is used as a feature. `RatingDeviation` and `NbPlays` are used only to *filter* rows (see below), never as model inputs.

## Data

`prepare.py` streams the ~300 MB zstd-compressed database (about 6.16M puzzles) and reservoir-samples 500,000 rows (seed 42) without decompressing the whole file to disk.

The label is only as good as the rating estimate, so the script keeps only puzzles with **`RatingDeviation ≤ 90` and `NbPlays ≥ 100`**. That drops noisy, rarely played puzzles and leaves **330,631**. They are split 80/20 into 264,504 training and 66,127 test puzzles.

## Features

All 15 features are computed with `python-chess` from the position the solver actually faces, after the opponent's setup move has been played.

| Group | Feature | Meaning |
|---|---|---|
| Solution | `n_solver_moves` | Number of moves the solver must find |
| | `frac_forcing` | Share of solver moves that are checks or captures |
| Key move (first solver move) | `key_is_capture`, `key_is_check`, `key_is_quiet`, `key_is_promotion` | What kind of move it is |
| | `key_piece_type` | Piece that moves (categorical: pawn … king) |
| | `key_is_sacrifice` | The moved piece lands where a lower-value enemy piece attacks it |
| Board | `total_material`, `material_balance` | Material on the board, and the solver's material edge |
| | `n_legal_moves` | Branching factor: how many candidate moves the solver has |
| | `n_attacked_pieces` | Non-king pieces under attack (tactical "tension") |
| | `white_to_move` | Side to move |
| Context | `game_phase` | 1.0 = full opening material, 0.0 = pawns and kings |
| | `has_opening_tag` | Lichess tags an opening only when the puzzle starts early in the game |

The features encode human intuitions about difficulty. Long lines are harder. Forcing moves (checks and captures) are easy to spot. Quiet moves and sacrifices are hard to spot. Busy positions with many options are harder to calculate.

## Models

`train.py` fits three models on the same split and evaluates each once on the test set:

1. **Baseline**: always predicts the mean training rating. Any useful model has to beat this.
2. **Ridge regression**: standardised numeric features plus a one-hot `key_piece_type`, with `RidgeCV` choosing alpha from 12 values between 10⁻² and 10³. It shows how much a linear model can extract.
3. **LightGBM** (the final model): a grid search over `num_leaves` ∈ {15, 31, 63}, `learning_rate` ∈ {0.03, 0.1} and `min_child_samples` ∈ {20, 100}, scored on a 15% validation slice of the training set with early stopping (patience 100, up to 3,000 trees). The winning configuration is refit on the full training set with the chosen number of trees. `key_piece_type` is handled as a native categorical feature.

Selected hyperparameters: `num_leaves=63`, `learning_rate=0.03`, `min_child_samples=100`, `n_estimators=902`. With 33K puzzles the grid picked the smallest trees (15 leaves). With 10× more data it picks the largest, because there are now enough examples to support finer feature interactions without overfitting.

The fitted booster is saved to `artifacts/model.txt` and can be loaded with `lightgbm.Booster(model_file=...)`.

## Results

Test set, 66,127 puzzles:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (mean) | 462.9 | 383.4 | 0.00 |
| Ridge | 368.7 | 295.8 | 0.37 |
| **LightGBM** | **348.2** | **279.1** | **0.43** |

- 43% of LightGBM's predictions fall within ±200 rating points of the true rating, and 75% fall within ±400.
- Ridge already captures most of the gain. Moving to trees adds a further ~17 points of MAE by modelling non-linear effects and interactions between features.

### Effect of 10× more data

I first trained the model on a 50,000-puzzle sample (33,209 after filtering). It reached an MAE of 284.7 on its own 6,642-puzzle test set. I then increased the sample to 500,000 puzzles (330,631 after filtering) and retrained. The two runs have different test sets, so for a fair comparison I trained on growing subsets of the new training set and scored each model on the same 66,127-puzzle test set:

| Training puzzles | LightGBM MAE | LightGBM RMSE |
|---|---|---|
| 26,567 (original size and config) | 284.5 | 354.2 |
| 66,000 | 282.0 | 352.0 |
| 132,000 | 280.2 | 349.6 |
| **264,504 (current)** | **279.1** | **348.2** |

Using 10× more data improved MAE by only about 5 rating points (≈2%), and each doubling helped less than the one before. The model is now limited by what the 15 features can express, not by how many examples it sees. Further accuracy will have to come from better features, not more data.

### Error by rating band

| Actual rating | Puzzles | MAE | Mean error (pred − actual) |
|---|---|---|---|
| < 1000 | 10,508 | 328 | +323 |
| 1000–1500 | 24,864 | 230 | +152 |
| 1500–2000 | 20,355 | 224 | −120 |
| 2000–2500 | 9,440 | 429 | −425 |
| 2500+ | 960 | 712 | −712 |

The model is most accurate in the 1000–2000 range, where most puzzles sit. At the extremes it **regresses toward the mean**: it overrates easy puzzles and badly underrates very hard ones. Its predictions have a standard deviation of 306, against 463 for the true ratings. The features can tell that a puzzle is "harder than average", but they cannot separate a 2,200 puzzle from a 2,700 one. What makes elite puzzles hard (a deep quiet move, a defensive resource) is mostly invisible to shallow board statistics.

## What the model learned

Global feature importance is the mean |SHAP| on the test set, in rating points. The direction column is read from the correlation between each feature's value and its SHAP contribution.

| Feature | Mean \|SHAP\| | Direction |
|---|---|---|
| `frac_forcing` | 172 | More checks and captures → **easier** |
| `n_solver_moves` | 119 | Longer solution → **harder** |
| `game_phase` | 51 | More pieces on the board → harder |
| `n_attacked_pieces` | 49 | More pieces under attack → harder |
| `total_material` | 45 | (strongly correlated with `game_phase`, so read the two together) |
| `key_is_check` | 39 | Key move is a check → easier |
| `n_legal_moves` | 35 | More candidate moves → harder |
| `key_piece_type` | 29 | — |
| `key_is_capture` | 15 | Key move is a capture → easier |
| `has_opening_tag` | 12 | Early-game puzzle → easier |
| `key_is_sacrifice` | 10 | Sacrifice → harder |

The two solution-line features dominate. The solution line alone carries much of the signal: in the data, one-move puzzles average about 1,020 and four-move puzzles about 1,880. Every top feature points in the direction a chess player would expect, which is a good sign that the model is learning real structure and not artefacts. `key_is_promotion` has almost no importance (0.4), because promotions as key moves are rare.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

On first launch the app finds no `artifacts/`, so it runs `prepare.py` and then `train.py`. This downloads the database once and takes about 5 minutes. You can also run the steps yourself:

```bash
.venv/bin/python prepare.py
.venv/bin/python train.py
```

The Streamlit dashboard shows the metrics table, a predicted-vs-actual scatter plot, MAE by rating band, SHAP importances, and a puzzle explorer. The explorer renders any test puzzle's board and solution next to its actual and predicted rating.

| File | Purpose |
|---|---|
| `prepare.py` | Download → sample → clean → features → `data/features.parquet` |
| `train.py` | Fit the three models and write `metrics.json`, `predictions.parquet`, `importance.csv` and `model.txt` to `artifacts/` |
| `app.py` | Streamlit dashboard |
