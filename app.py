"""
app.py — phone-friendly web front end for fib_ladder.py

Local test:   pip install streamlit yfinance pandas
              streamlit run app.py
Deploy free:  push app.py, fib_ladder.py, requirements.txt to a GitHub repo,
              then https://share.streamlit.io -> New app -> pick the repo -> Deploy.
              You get a public URL; bookmark it on the phone's home screen.
"""
import math
import streamlit as st
import fib_ladder as fl

st.set_page_config(page_title="Fib Ladder", page_icon="📐", layout="centered")
st.title("Fib Ladder stock report")
st.caption("Reverse-engineered template, not the academy's method, not financial advice.")

ticker = st.text_input("Ticker", placeholder="e.g. QCOM").strip().upper()

with st.expander("Advanced (optional)"):
    zz = st.slider("Swing sensitivity (ZigZag %)", 3, 15, 8,
                   help="Smaller = picks smaller, more recent swings") / 100
    manual = st.checkbox("Set the swing anchors myself")
    hi = lo = None
    if manual:
        c1, c2 = st.columns(2)
        hi = c1.number_input("Swing high (1.0 line)", min_value=0.0, value=0.0)
        lo = c2.number_input("Swing low (0 line)", min_value=0.0, value=0.0)
        if not hi or not lo:
            hi = lo = None

run = st.button("Get report", type="primary", use_container_width=True)


def r0(x):
    return int(round(x))


if run and ticker:
    fl.ZIGZAG_PCT = zz
    try:
        with st.spinner(f"Fetching {ticker}…"):
            res = fl.analyse(ticker, hi, lo)
    except Exception as e:
        st.error(f"Could not analyse {ticker}: {e}")
        st.stop()

    L = res["ladder"]
    boa = L["BOA (entry / 1.0 line)"]
    sl1, sl2 = L["SL1 (first stop)"], L["SL2 (deep stop / 0 line)"]
    t1, t2, t3 = L["Target 1"], L["Target 2"], L["Target 3"]
    cmp_ = res["cmp"]

    st.subheader(f"{res['ticker']} – Stock Report")
    st.text(
        f"CMP: ${cmp_:.2f}   (data as of {res['asof']})\n"
        f"BOA: {r0(boa)}\n"
        f"Targets: {r0(t1)}, {r0(t2)}, {r0(t3)}\n"
        f"SL: {r0(sl1)} or {r0(sl2)}\n"
        f"Period: {res['period']}"
    )

    if cmp_ < boa:
        st.info(
            f"Price is below BOA — 'accumulate on dips' format.\n\n"
            f"Buying zone near the swing low ≈ {r0(math.ceil(sl2 / 5) * 5)}; "
            f"deeper buys {r0(sl2 - res['D'] / 3)} and {r0(sl2 - res['D'])}.\n\n"
            f"A close above {r0(boa)} for 3 sessions would be the breakout confirmation."
        )
    else:
        st.info(f"Price is above BOA — buy-on-approach format (entry near {r0(boa)}).")

    st.markdown("**Full ladder**")
    rows = [{"Multiplier (×D)": f"{m:.3f}", "Level": n, "Price": f"${L[n]:,.2f}"}
            for n, m in fl.MULTIPLIERS.items()]
    st.table(rows)
    st.caption(f"Anchor: {res['anchor_note']} · D = {res['D']:.2f} "
               f"({res['D_pct']:.0%} of BOA). If this swing isn't the one you'd draw, "
               f"change the sensitivity or set the anchors under Advanced.")
