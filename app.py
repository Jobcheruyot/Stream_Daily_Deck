
import io, os, re, types, textwrap
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Notebook Runner (Non-Starred)", page_icon="🧭", layout="wide")

st.sidebar.header("Upload")
csv_file = st.sidebar.file_uploader("CSV (up to ~500MB)", type=["csv"])
py_file = st.sidebar.file_uploader("Notebook .py", type=["py"])

default_py_path = Path("/mnt/data/superdeck.py")
source_text = None
if py_file is not None:
    source_text = py_file.read().decode("utf-8", errors="ignore")
elif default_py_path.exists():
    source_text = default_py_path.read_text(encoding="utf-8", errors="ignore")
else:
    st.stop()

# Load CSV once
df_user = None
if csv_file is not None:
    df_user = pd.read_csv(csv_file, low_memory=False)

# ---------------- Parsing ----------------
def parse_blocks(text):
    blocks = []
    cur_title = None
    cur_code = []
    heading_re = re.compile(r'^\s*("""|\'\'\')\s*#\s*(.*?)\s*(?:\1)\s*$')
    md_heading_re = re.compile(r'^\s*#\s*([#\s]*)(.*)$')
    lines = text.splitlines()
    i = 0
    def push():
        nonlocal cur_title, cur_code, blocks
        if cur_title is not None or cur_code:
            code_str = "\n".join(cur_code).strip()
            if code_str:
                blocks.append((cur_title or "Untitled", code_str))
        cur_title, cur_code = None, []
    while i < len(lines):
        ln = lines[i]
        m = heading_re.match(ln)
        if m:
            push()
            cur_title = m.group(2).strip()
            i += 1
            continue
        m2 = re.match(r'^\s*("""|\'\'\')\s*(.*?)\s*(?:\1)\s*$', ln)
        if m2 and (i+1 < len(lines)) and lines[i+1].strip()=="" and (m2.group(2).lstrip().startswith("#") or m2.group(2).lstrip().startswith("##")):
            title = re.sub(r'^\s*#+\s*', '', m2.group(2)).strip()
            push()
            cur_title = title
            i += 1
            continue
        if md_heading_re.match(ln) and (i==0 or lines[i-1].strip()=="") and (i+1==len(lines) or lines[i+1].strip()==""):
            title = md_heading_re.match(ln).group(2).strip()
            if title:
                push()
                cur_title = title
                i += 1
                continue
        cur_code.append(ln)
        i += 1
    push()
    out = []
    for title, code in blocks:
        t = re.sub(r'^\s*#+\s*', '', title or '').strip()
        out.append((t, code))
    return out

blocks = parse_blocks(source_text)

# Filter non-starred
visible_blocks = [(t,c) for (t,c) in blocks if not t.strip().startswith("*")]

if not visible_blocks:
    st.error("No non-starred sections found.")
    st.stop()

# Sidebar menu
titles = [t for t,_ in visible_blocks]
choice = st.sidebar.radio("Sections", options=titles, index=0)
run_all = st.sidebar.button("▶ Run ALL Non-Starred")

# Helper: patch plotting/display so .show() draws in Streamlit
def setup_exec_env(df):
    glb = {}
    glb.update({'pd': pd, 'np': np, 'px': px, 'go': go, 'plt': plt, 'display': st.write})
    if df is not None:
        def _read_csv_hook(*args, **kwargs):
            return df.copy()
        glb['__orig_read_csv'] = pd.read_csv
        pd.read_csv = _read_csv_hook
        glb['_restore'] = lambda: setattr(pd, 'read_csv', glb['__orig_read_csv'])
    else:
        glb['_restore'] = lambda: None
    def _fig_show(self, *a, **k):
        st.plotly_chart(self, use_container_width=True)
    go.Figure.show = _fig_show
    def _plt_show(*a, **k):
        st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
    plt.show = _plt_show
    return glb

def execute_block(title, code, df):
    st.markdown(f"### {title}")
    with st.expander("Show code", expanded=False):
        st.code(code, language="python")
    glb = setup_exec_env(df)
    try:
        exec(compile(code, filename=f"<block:{title}>", mode="exec"), glb, glb)
    except Exception as e:
        st.exception(e)
    finally:
        glb['_restore']()

st.title("Notebook Runner — Only Non-Starred Sections")
st.caption("Runs the code blocks whose headings do not begin with '*' from your .py export.")

if run_all:
    for t,c in visible_blocks:
        execute_block(t, c, df_user)
else:
    for t,c in visible_blocks:
        if t == choice:
            execute_block(t, c, df_user)
            break
