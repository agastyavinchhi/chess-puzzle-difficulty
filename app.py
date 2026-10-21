"""Streamlit dashboard for the chess puzzle difficulty models.

Run with: streamlit run app.py
"""

import json

import chess
import chess.svg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import prepare
import train

ARTIFACTS = train.ARTIFACTS
REQUIRED = ["metrics.json", "predictions.parquet", "importance.csv", "model.txt"]
BANDS = [0, 1000, 1500, 2000, 2500, np.inf]
BAND_LABELS = ["<1000", "1000–1500", "1500–2000", "2000–2500", "2500+"]

st.set_page_config(page_title="Chess Puzzle Difficulty", layout="wide")
st.title("Chess puzzle difficulty")
st.caption(
    "Predicting a Lichess puzzle's rating from its board and solution only "
    "(no popularity, play counts, rating deviation, or themes)."
)

if not all((ARTIFACTS / f).exists() for f in REQUIRED):
    with st.status("Artifacts missing; building them (download, features, training)...") as s:
        if not prepare.FEATURES_PATH.exists():
            st.write("Preparing data (downloads ~300 MB on first run)...")
            prepare.main()
        st.write("Training models...")
        train.main()
        s.update(label="Artifacts built", state="complete")


@st.cache_data
def load():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    preds = pd.read_parquet(ARTIFACTS / "predictions.parquet")
    importance = pd.read_csv(ARTIFACTS / "importance.csv")
    return metrics, preds, importance


metrics, preds, importance = load()

# 1. Metrics table
st.header("Test-set metrics")
table = pd.DataFrame(metrics["models"]).T.round({"RMSE": 1, "MAE": 1, "R2": 2})
st.dataframe(table)
st.caption(f"{metrics['n_train']:,} training / {metrics['n_test']:,} test puzzles.")

left, right = st.columns(2)

# 2. Predicted vs. actual
with left:
    st.header("Predicted vs. actual (LightGBM)")
    lo = min(preds["actual"].min(), preds["predicted"].min())
    hi = max(preds["actual"].max(), preds["predicted"].max())
    fig = px.scatter(
        preds, x="actual", y="predicted", opacity=0.25, hover_data=["id"],
        labels={"actual": "Actual rating", "predicted": "Predicted rating"},
    )
    fig.update_traces(marker=dict(size=4))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color="red", dash="dash")))
    st.plotly_chart(fig, width="stretch")

# 3. MAE by rating band
with right:
    st.header("MAE by rating band")
    banded = preds.assign(band=pd.cut(preds["actual"], BANDS, labels=BAND_LABELS, right=False))
    by_band = (
        banded.assign(
            LightGBM=(banded["predicted"] - banded["actual"]).abs(),
            Ridge=(banded["predicted_ridge"] - banded["actual"]).abs(),
        )
        .groupby("band", observed=False)
        .agg(LightGBM=("LightGBM", "mean"), Ridge=("Ridge", "mean"), n=("actual", "size"))
        .reset_index()
    )
    fig = px.bar(
        by_band.melt(id_vars=["band", "n"], value_vars=["LightGBM", "Ridge"],
                     var_name="model", value_name="MAE"),
        x="band", y="MAE", color="model", barmode="group", hover_data=["n"],
        labels={"band": "Actual rating band"},
    )
    st.plotly_chart(fig, width="stretch")

# 4. SHAP importance
st.header("Feature importance (mean |SHAP|, LightGBM)")
fig = px.bar(
    importance.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
    orientation="h", labels={"mean_abs_shap": "Mean |SHAP| (rating points)", "feature": ""},
)
fig.update_layout(height=450)
st.plotly_chart(fig, width="stretch")

# 5. Puzzle explorer
st.header("Puzzle explorer")
puzzle_id = st.selectbox("Test puzzle", preds["id"], index=0)
row = preds.set_index("id").loc[puzzle_id]

board = chess.Board(row["FEN"])
uci = row["moves"].split()
setup = chess.Move.from_uci(uci[0])
board.push(setup)

# Solution line in SAN
san, line = [], board.copy()
for mv in uci[1:]:
    move = chess.Move.from_uci(mv)
    san.append(line.san(move))
    line.push(move)

c1, c2 = st.columns([1, 1])
with c1:
    svg = chess.svg.board(board, lastmove=setup, orientation=board.turn, size=420)
    st.image(svg, width=420)
with c2:
    side = "White" if board.turn == chess.WHITE else "Black"
    st.markdown(f"**{side} to move.** Opponent just played `{uci[0]}`.")
    st.markdown("Solution: " + " ".join(
        f"**{m}**" if i % 2 == 0 else m for i, m in enumerate(san)
    ))
    st.caption("Solver moves in bold.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Actual rating", int(row["actual"]))
    m2.metric("Predicted (LightGBM)", int(round(row["predicted"])),
              delta=int(round(row["predicted"] - row["actual"])), delta_color="off")
    m3.metric("Predicted (Ridge)", int(round(row["predicted_ridge"])))
    st.markdown(f"[View on Lichess](https://lichess.org/training/{puzzle_id})")
