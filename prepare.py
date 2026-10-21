"""Download the Lichess puzzle DB, sample 500K rows, clean, and compute features.

Output: data/features.parquet
"""

import csv
import io
import random
import shutil
import sys
import urllib.request
from pathlib import Path

import chess
import pandas as pd
import zstandard

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "lichess_db_puzzle.csv.zst"
SAMPLE_PATH = DATA_DIR / "sample.csv"
FEATURES_PATH = DATA_DIR / "features.parquet"

SAMPLE_SIZE = 500_000
SEED = 42
MAX_RATING_DEVIATION = 90
MIN_NB_PLAYS = 100

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
# Non-pawn material weights for the game-phase score (24 = full starting set).
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24

# Only these columns are ever used as model inputs. Popularity, NbPlays,
# RatingDeviation, Themes and DailyDate are leaky and deliberately excluded.
FEATURES = [
    "n_solver_moves",
    "frac_forcing",
    "key_is_capture",
    "key_is_check",
    "key_is_quiet",
    "key_is_promotion",
    "key_piece_type",
    "key_is_sacrifice",
    "total_material",
    "material_balance",
    "n_legal_moves",
    "n_attacked_pieces",
    "white_to_move",
    "game_phase",
    "has_opening_tag",
]
CATEGORICAL = ["key_piece_type"]


def download():
    if RAW_PATH.exists():
        print(f"Using cached {RAW_PATH}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    tmp = RAW_PATH.with_suffix(".part")
    print(f"Downloading {URL} (~300 MB)...")
    with urllib.request.urlopen(URL) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(RAW_PATH)


def sample():
    """Reservoir-sample SAMPLE_SIZE rows while stream-decompressing the file."""
    if SAMPLE_PATH.exists():
        print(f"Using cached {SAMPLE_PATH}")
        return pd.read_csv(SAMPLE_PATH, keep_default_na=False)
    rng = random.Random(SEED)
    reservoir = []
    with open(RAW_PATH, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        rows = csv.reader(text)
        header = next(rows)
        for i, row in enumerate(rows):
            if i < SAMPLE_SIZE:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < SAMPLE_SIZE:
                    reservoir[j] = row
        print(f"Streamed {i + 1:,} rows, sampled {len(reservoir):,}")
    df = pd.DataFrame(reservoir, columns=header)
    df.to_csv(SAMPLE_PATH, index=False)
    return pd.read_csv(SAMPLE_PATH, keep_default_na=False)


def clean(df):
    n = len(df)
    df = df[(df["RatingDeviation"] <= MAX_RATING_DEVIATION) & (df["NbPlays"] >= MIN_NB_PLAYS)]
    print(
        f"Cleaning: kept {len(df):,} of {n:,} rows, dropped {n - len(df):,} "
        f"(RatingDeviation <= {MAX_RATING_DEVIATION}, NbPlays >= {MIN_NB_PLAYS})"
    )
    return df.reset_index(drop=True)


def material(board, color):
    return sum(
        PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color
    )


def featurize(fen, moves, opening_tags):
    board = chess.Board(fen)
    uci = moves.split()
    board.push_uci(uci[0])  # opponent's setup move; board is now what the solver sees
    solver = board.turn
    puzzle_board = board.copy()

    # Solution line: solver moves are uci[1], uci[3], ...
    line = board.copy()
    solver_moves, forcing = 0, 0
    for k, mv in enumerate(uci[1:]):
        move = chess.Move.from_uci(mv)
        if k % 2 == 0:
            solver_moves += 1
            forcing += line.is_capture(move) or line.gives_check(move)
        line.push(move)

    # Key move
    key = chess.Move.from_uci(uci[1])
    piece = board.piece_at(key.from_square)
    is_capture = board.is_capture(key)
    is_check = board.gives_check(key)
    is_promotion = key.promotion is not None
    after = board.copy()
    after.push(key)
    moved_type = after.piece_at(key.to_square).piece_type
    attackers = after.attackers(not solver, key.to_square)
    is_sacrifice = any(
        PIECE_VALUES[after.piece_at(sq).piece_type] < PIECE_VALUES[moved_type]
        and after.piece_at(sq).piece_type != chess.KING
        for sq in attackers
    )

    # Board
    own, opp = material(puzzle_board, solver), material(puzzle_board, not solver)
    attacked = sum(
        1
        for sq, p in puzzle_board.piece_map().items()
        if p.piece_type != chess.KING and puzzle_board.is_attacked_by(not p.color, sq)
    )
    phase_score = sum(
        PHASE_WEIGHTS.get(p.piece_type, 0) for p in puzzle_board.piece_map().values()
    )

    return {
        "n_solver_moves": solver_moves,
        "frac_forcing": forcing / solver_moves,
        "key_is_capture": int(is_capture),
        "key_is_check": int(is_check),
        "key_is_quiet": int(not (is_capture or is_check or is_promotion)),
        "key_is_promotion": int(is_promotion),
        "key_piece_type": piece.piece_type,
        "key_is_sacrifice": int(is_sacrifice),
        "total_material": own + opp,
        "material_balance": own - opp,
        "n_legal_moves": puzzle_board.legal_moves.count(),
        "n_attacked_pieces": attacked,
        "white_to_move": int(solver == chess.WHITE),
        # 1.0 = full opening material, 0.0 = bare kings and pawns
        "game_phase": min(phase_score, MAX_PHASE) / MAX_PHASE,
        "has_opening_tag": int(bool(opening_tags.strip())),
    }


def main():
    download()
    df = clean(sample())
    feats = pd.DataFrame(
        [featurize(f, m, o) for f, m, o in zip(df["FEN"], df["Moves"], df["OpeningTags"])]
    )
    out = pd.concat([df[["PuzzleId", "FEN", "Moves", "Rating"]], feats], axis=1)
    out.to_parquet(FEATURES_PATH, index=False)
    print(f"Wrote {len(out):,} rows x {len(FEATURES)} features to {FEATURES_PATH}")


if __name__ == "__main__":
    sys.exit(main())
