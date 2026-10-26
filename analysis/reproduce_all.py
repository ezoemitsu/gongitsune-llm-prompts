#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_all.py
================
s2026-023（JSiSE 条件付採録）改稿用：原稿に載せる全数値を1本で再現・確定するスクリプト。

- プロジェクトのデータは読み込みのみ。書き込みは本スクリプトと同じフォルダの out/ と
  数値確定表.md（本スクリプトが自動生成）のみ。
- 乱数シード 20261017（各節で SEED + 節番号 を親シードとして使う）。
- SOLO は既定で既存 Claude 評価（144記述。ルーブリックの148記述と結合して143記述）を暫定使用。
  3LLM 再評価の結果が得られたら --solo reeval --solo-file <CSV> で差し替えて再実行する。

実行（作業ディレクトリはどこでもよい）:
  python3 reproduce_all.py
  python3 reproduce_all.py --solo reeval --solo-file ../solo_reeval/out/SOLO_3LLM評価結果.csv
  python3 reproduce_all.py --fast      # 再標本化回数を1/10にした動作確認（数値は確定値ではない）
"""
import argparse
import datetime as dt
import itertools
import json
import math
import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, cohen_kappa_score,
                             davies_bouldin_score, f1_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 120)

# =============================================================================
# 0. 設定
# =============================================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # .../jsise_tokushu27
OUT = HERE / "out"
NUMBER_TABLE = HERE / "数値確定表.md"
SEED = 20261017

FEATS = ["view_count", "view_duration", "view_mean_duration", "view_breadth", "view_depth", "view_gini"]
FEATS_JP = ["閲覧回数", "閲覧時間", "平均閲覧時間/回", "参照の広さ", "参照の深さ", "参照の集中度"]
GROUP_MAP = {"高頻度探索型": "広範・高頻度群", "バランス型": "広範・中頻度群", "選択的熟考型": "限定・長時間群"}
OLD_NAME = {"広範・高頻度群": "網羅的探索群", "広範・中頻度群": "均等群", "限定・長時間群": "局所滞留群"}
ORDER = ["広範・高頻度群", "広範・中頻度群", "限定・長時間群"]
R4 = {"S": 4, "A": 3, "B": 2, "C": 1}
R3 = {"A": 3, "B": 2, "C": 1}
INV4 = {4: "S", 3: "A", 2: "B", 1: "C"}
INV3 = {3: "A", 2: "B", 1: "C"}
PAGE_SCENE = {4: "第1場面", 5: "第2場面", 6: "第3場面", 7: "第4場面", 8: "第5場面", 9: "第6場面"}
STORY_NAMES = {"兵十", "加助", "弥助", "新兵衛"}

V4 = {  # 原稿v4（投稿版）の数値。旧→新の対照に使う
    "T7_KW": "6指標のKW（H = 18.88, 15.29, 11.83, 12.98, 18.00, 4.49；ε² = .675〜.100）",
    "T8_pop": "全148サンプルから層化無作為抽出した50サンプル（33.8%）",
    "T9_row2": "p9段階：S 2，A 12，B 9，C 5（n = 28）",
    "T11_KW_mean": "H(2) = 3.30, p = .192, ε² = .052",
    "T11_KW_others": "すべて p > .15（変化量はCSVで p = .639）",
    "T12": "χ²(2) = 5.67, p = .059, Cramér's V = .450（最終ページp9）",
    "T13_hi": "W = 4.0, p = .343, r = .387",
    "T13_mid": "W = 10.0, p = .008, r = .629",
    "T13_lim": "W = 0.0, p = .109, r = .802",
    "T14": "参照の広さ +0.844 ほか6係数（L2ロジスティック回帰）",
    "abs_climax": "at the climax ... 67% vs 17%/25% (χ²(2) = 5.67, p = .059, V = .450)",
    "abs_within": "significant within-group deepening only here (Δ = +0.50, p = .008)",
    "s422": "クライマックス場面（p9）で単元目標を達成した児童は14名（50.0%）",
    "s431_consistent": "達成判定の他のパターンでも均等群の優位が一貫して観察された",
    "s432": "均等群の達成12名に①②③の3要素が共通",
    "s5_pred": "参照の「広さ」が達成の主たる予測因子",
    "gini_def": "ジニ係数は分母を27（自分以外の児童数）に固定",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def H1(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ------------------------------------------------------------------ 書式
def isnan(x):
    try:
        return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))
    except Exception:
        return False


def _r(x, d):
    """四捨五入（round-half-up）した文字列。Pythonの書式は偶数丸め（0.3125→0.312）のため使わない"""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def fp(p):
    """APA形式の p 値（先頭の0を省く。.001未満は '< .001'）"""
    if isnan(p):
        return "—"
    if p < .001:
        return "< .001"
    s = _r(p, 3)
    if s.startswith("1"):
        return "1.000"
    return s[1:] if s.startswith("0") else s


def fpe(p):
    """本文用：'= .069' / '< .001'"""
    s = fp(p)
    return s if s.startswith("<") or s == "—" else "= " + s


def fz(x, d=3):
    """上限1の統計量（r, V, κ, ε², ARI, Jaccard）：先頭の0を省く"""
    if isnan(x):
        return "—"
    s = _r(x, d)
    neg = s.startswith("-")
    s = s.lstrip("-")
    if s.startswith("0."):
        s = s[1:]
    if neg and float(s) != 0:
        s = "−" + s
    return s


def fn(x, d=2):
    if isnan(x):
        return "—"
    s = _r(x, d)
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−")


def fs(x, d=2):
    if isnan(x):
        return "—"
    s = _r(x, d)
    if float(s) == 0:
        return "+" + s.lstrip("-")
    return ("+" + s if not s.startswith("-") else s).replace("-", "−")


def frac(a, n, d=1):
    if n == 0:
        return "—"
    return f"{a}/{n} ({_r(100 * a / n, d)}%)"


def msd(s, d=2):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "—"
    if len(s) == 1:
        return f"{fn(s.mean(), d)} (—)"
    return f"{fn(s.mean(), d)} ({fn(s.std(ddof=1), d)})"


def msd_sign(s, d=2):
    s = pd.Series(s).dropna()
    return f"{fs(s.mean(), d)} ({fn(s.std(ddof=1), d)})"


def mask_names(text):
    """記述中の児童の姓（漢字2〜4字＋さん/くん/君/ちゃん）を〇〇に置換。物語の登場人物名は除く"""
    if not isinstance(text, str):
        return text

    def rep(m):
        return m.group(0) if m.group(1) in STORY_NAMES else "〇〇"
    return re.sub(r"([一-鿿豈-﫿々]{2,4})(?=さん|くん|君|ちゃん)", rep, text)


def md_table(header, rows):
    esc = lambda x: str(x).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(x) for x in r) + " |")
    return "\n".join(out)


# =============================================================================
# 1. データ読込
# =============================================================================
def load_cluster():
    c = pd.read_csv(ROOT / "05_クラスタリング結果/clustering_results.csv")
    c = c.rename(columns={"student": "child"})
    c["group"] = c["cluster_name"].map(GROUP_MAP)
    assert c["group"].notna().all() and len(c) == 28
    return c


def load_rubric():
    d = pd.read_csv(ROOT / "03_LLM評価データ/単元目標準拠ルーブリック_3LLM評価結果.csv", encoding="utf-8-sig")
    d = d.rename(columns={"児童ID": "child", "ページ": "page"})
    d["page"] = d["page"].astype(int)
    d["total"] = d["3LLM中央値_総合"].map(R4)
    d["eA"] = d["3LLM中央値_要素A_本文根拠"].map(R3)
    d["eB"] = d["3LLM中央値_要素B_心情読み取り"].map(R3)
    d["eC"] = d["3LLM中央値_要素C_解釈統合"].map(R3)
    # 要素1・2による再構成（規則版）：両方A→A(3)、いずれかC→C(1)、それ以外B(2)
    d["AB_rule"] = np.select([(d.eA == 3) & (d.eB == 3), (d.eA == 1) | (d.eB == 1)], [3, 1], default=2)
    d["AB_mean"] = (d.eA + d.eB) / 2
    assert len(d) == 148 and d[["total", "eA", "eB", "eC"]].notna().all().all()
    return d


def load_text():
    t = pd.read_csv(ROOT / "03_LLM評価データ/振り返り記述_全データ_long形式.csv", encoding="utf-8-sig")
    t = t.rename(columns={"児童ID": "child", "ページ": "page", "振り返り記述": "text", "文字数": "nchar"})
    t["page"] = t["page"].astype(int)
    return t[["child", "page", "text", "nchar"]]


def load_raw():
    r = pd.read_csv(ROOT / "03_LLM評価データ/単元目標準拠ルーブリック_3LLM評価_raw.csv", encoding="utf-8-sig")
    r = r.rename(columns={"児童ID": "child", "ページ": "page"})
    r["page"] = r["page"].astype(int)
    return r


def load_solo_claude():
    s = pd.read_csv(ROOT / "分析データ/SOLO/claude)solo_scoring_by_page.csv")
    s["page"] = s["page"].astype(str).str.replace("p", "").astype(int)
    s = s.rename(columns={"uid": "child", "Score": "solo"})
    return s[["child", "page", "solo"]]


def load_solo(mode, path):
    if mode == "claude":
        return load_solo_claude(), "SOLO（既存Claude評価・暫定）", "既存Claude評価（単一モデル・1回採点、144記述）"
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    s = pd.read_csv(p, encoding="utf-8-sig")
    s = s.rename(columns={"児童ID": "child", "ページ": "page", "3LLM中央値_SOLO": "solo"})
    s["page"] = s["page"].astype(int)
    s = s.dropna(subset=["solo"])
    s["solo"] = s["solo"].astype(int)
    return s[["child", "page", "solo"]], "SOLO（中立基準・3LLM再評価）", f"中立基準3LLM再評価（{p.name}）"


# =============================================================================
# 2. 児童単位の集約と検定関数
# =============================================================================
RND = 6   # 決定事項：差・集約値は小数第6位で丸めてから同順位処理


def aggregate(pg, col, thr, final_page, alt_last=False):
    """児童ごとの集約。final_page: {child: 最終振り返りのページ}（ルーブリック148記述で定義）。
    alt_last=True のとき、最終振り返りの値が欠損なら直前の記述の値で代替する（SOLOの感度分析用）。"""
    rows = []
    for ch, g in pg.groupby("child"):
        g = g.dropna(subset=[col]).sort_values("page")
        if len(g) == 0:
            continue
        v = g[col].to_numpy(float)
        front = g.loc[g.page.isin([4, 5, 6]), col].to_numpy(float)
        back = g.loc[g.page.isin([7, 8, 9]), col].to_numpy(float)
        fpg = final_page[ch]
        lastv = g.loc[g.page == fpg, col]
        if len(lastv):
            last = float(lastv.iloc[0]); last_src = fpg
        elif alt_last:
            last = float(v[-1]); last_src = int(g.page.iloc[-1])
        else:
            last = np.nan; last_src = np.nan
        p9 = g.loc[g.page == 9, col]
        p9 = float(p9.iloc[0]) if len(p9) else np.nan
        fm = front.mean() if len(front) else np.nan
        bm = back.mean() if len(back) else np.nan
        rows.append(dict(
            child=ch, n_rec=len(v), mean=round(v.mean(), RND), max=v.max(), last=last, last_page=last_src,
            p9=p9, front=round(fm, RND) if not isnan(fm) else np.nan, back=round(bm, RND) if not isnan(bm) else np.nan,
            change=round(bm - fm, RND) if not (isnan(fm) or isnan(bm)) else np.nan,
            ach_major=int((v >= thr).mean() >= 0.5),
            ach_last=(np.nan if isnan(last) else int(last >= thr)),
            ach_any=int(v.max() >= thr),
            ach_front=(np.nan if len(front) == 0 else int((front >= thr).mean() >= 0.5)),
            ach_back=(np.nan if len(back) == 0 else int((back >= thr).mean() >= 0.5)),
            ach_p9=(np.nan if isnan(p9) else int(p9 >= thr)),
        ))
    return pd.DataFrame(rows)


def kw_test(df, col, group="group"):
    gs = [df.loc[df[group] == g, col].dropna().to_numpy(float) for g in ORDER]
    n = sum(len(x) for x in gs)
    if any(len(x) == 0 for x in gs) or len(np.unique(np.concatenate(gs))) < 2:
        return dict(H=np.nan, p=np.nan, eps2=np.nan, eta2H=np.nan, n=n)
    H, p = stats.kruskal(*gs)
    k = 3
    return dict(H=H, p=p, eps2=H / (n - 1), eta2H=(H - k + 1) / (n - k), n=n)


def kw_perm(df, col, rng, B=20000, group="group"):
    d = df.dropna(subset=[col])
    vals = d[col].to_numpy(float)
    labs = d[group].to_numpy()
    H0 = stats.kruskal(*[vals[labs == g] for g in ORDER])[0]
    cnt = 0
    for _ in range(B):
        pl = rng.permutation(labs)
        if stats.kruskal(*[vals[pl == g] for g in ORDER])[0] >= H0 - 1e-12:
            cnt += 1
    return (cnt + 1) / (B + 1)


def _tables_rx2(row_tot, c1):
    for a in itertools.product(*[range(0, n + 1) for n in row_tot]):
        if sum(a) == c1:
            yield np.array([[a[i], row_tot[i] - a[i]] for i in range(len(row_tot))])


def _logp(t):
    rows, cols, N = t.sum(1), t.sum(0), t.sum()
    lp = sum(math.lgamma(x + 1) for x in rows) + sum(math.lgamma(x + 1) for x in cols) - math.lgamma(N + 1)
    return lp - sum(math.lgamma(x + 1) for x in t.ravel())


def ffh(t):
    """Fisher–Freeman–Halton 正確検定（r×2）：観測表以下の確率をもつ表の確率和"""
    t = np.asarray(t, int)
    lp0 = _logp(t)
    p = sum(math.exp(_logp(tt)) for tt in _tables_rx2(t.sum(1), t.sum(0)[0]) if _logp(tt) <= lp0 + 1e-7)
    return min(p, 1.0)


def crosstab_test(df, col, group="group"):
    d = df.dropna(subset=[col])
    t = np.array([[int(((d[group] == g) & (d[col] == 1)).sum()), int(((d[group] == g) & (d[col] == 0)).sum())] for g in ORDER])
    res = dict(table=t.tolist(), n=int(t.sum()),
               rates=[(t[i, 0] / t[i].sum() if t[i].sum() else np.nan) for i in range(3)],
               counts=[(int(t[i, 0]), int(t[i].sum())) for i in range(3)])
    if (t.sum(0) == 0).any():
        res.update(chi2=np.nan, p_chi=np.nan, V=np.nan, p_ffh=1.0, min_exp=np.nan)
        return res
    chi2, p, _, exp = stats.chi2_contingency(t, correction=False)
    res.update(chi2=chi2, p_chi=p, V=math.sqrt(chi2 / t.sum()), p_ffh=ffh(t), min_exp=exp.min())
    return res


def wilcoxon_exact(change):
    """前半→後半のWilcoxon符号順位検定。入力は丸める前の（後半平均−前半平均）を小数第6位で丸めた差
    （前半・後半を別々に丸めてから引くと 0.333334 と 0.333333 のように同順位が崩れるため、必ず差で渡す）。
    ゼロ差を除外、同順位は平均順位。正確p は符号の全組合せ（2^n）の分布を動的計画法で求める。"""
    d = np.round(np.asarray(change, float), RND)
    N = len(d)
    nz = d[d != 0]
    n = len(nz)
    out = dict(N=N, n_nonzero=n, delta_M=d.mean() if N else np.nan, delta_SD=d.std(ddof=1) if N > 1 else np.nan,
               n_pos=int((nz > 0).sum()), n_neg=int((nz < 0).sum()))
    if n == 0:
        out.update(Tplus=0, Tminus=0, W=np.nan, p_exact=1.0, z=0.0, r=0.0, r_rb=np.nan, p_min=1.0)
        return out
    ranks = stats.rankdata(np.abs(nz))
    r2 = np.round(ranks * 2).astype(int)
    tot2 = int(r2.sum())
    dist = np.zeros(tot2 + 1, dtype=np.float64)
    dist[0] = 1.0
    for r in r2:
        new = dist.copy()
        new[r:] += dist[:tot2 + 1 - r]
        dist = new
    s = np.arange(tot2 + 1)
    mu2 = tot2 / 2
    tplus2 = int(r2[nz > 0].sum())
    obs = abs(tplus2 - mu2)
    p = dist[np.abs(s - mu2) >= obs - 1e-9].sum() / 2 ** n
    pmin = dist[np.abs(s - mu2) >= abs(tot2 - mu2) - 1e-9].sum() / 2 ** n
    Tp, Tm = tplus2 / 2, (tot2 - tplus2) / 2
    _, cnt = np.unique(np.abs(nz), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(c ** 3 - c for c in cnt) / 48
    z = (Tp - n * (n + 1) / 4) / math.sqrt(var) if var > 0 else 0.0
    out.update(Tplus=Tp, Tminus=Tm, W=min(Tp, Tm), p_exact=min(p, 1.0), z=z, r=abs(z) / math.sqrt(N),
               r_rb=(Tp - Tm) / (Tp + Tm), p_min=min(pmin, 1.0))
    return out


def map_labels(orig, lab, k):
    cm = np.zeros((k, k))
    for o, l in zip(orig, lab):
        cm[o, l] += 1
    r, c = linear_sum_assignment(-cm)
    mp = dict(zip(c, r))
    return np.array([mp.get(l, -1) for l in lab])


def km(X, k, seed, n_init=50):
    return KMeans(n_clusters=k, n_init=n_init, random_state=int(seed)).fit(X)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def gini_mad(x):
    x = np.asarray(x, float)
    n = len(x)
    return np.abs(x[:, None] - x[None, :]).sum() / (2 * n * n * x.mean())


def icc2k(M):
    n, k = M.shape
    gm = M.mean()
    ssr = k * ((M.mean(1) - gm) ** 2).sum()
    ssc = n * ((M.mean(0) - gm) ** 2).sum()
    sse = ((M - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (msc - mse) / n)


# =============================================================================
# 3. 各節
# =============================================================================
def sec_T2(c, S):
    H1("[表2] 6指標の定義・算出式と生ログからの再計算")
    xlsx = ROOT / "分析データ/logs_with_duration_v2.xlsx"
    csvp = ROOT / "分析データ/archive/logs_with_duration_v2.csv"
    L = pd.read_excel(xlsx, sheet_name="logs_with_duration_v2")
    Lc = pd.read_csv(csvp, encoding="utf-8-sig")

    def views(L):
        return L[(L.Action == "ew_view") & L.ew_id.astype(str).str.startswith("児童") & (L.ew_id != L.user_id)
                 & L.user_id.astype(str).str.startswith("児童")]

    def compute(L):
        v = views(L)
        rows = []
        for u, g in v.groupby("user_id"):
            cnts = g.ew_id.value_counts().to_numpy()
            cnt = len(g)
            dur = g.duration_sec.sum()
            br = g.ew_id.nunique()
            rows.append(dict(child=u, r_count=cnt, r_dur=dur, r_mdur=dur / cnt, r_breadth=br, r_depth=cnt / br,
                             r_gini=gini(cnts), r_gini_mad=gini_mad(cnts),
                             r_gini27=gini(np.concatenate([cnts, np.zeros(27 - br)]))))
        return pd.DataFrame(rows)
    R = compute(L)
    Rc = compute(Lc)
    same_src = bool(np.allclose(R.sort_values("child")[["r_count", "r_dur", "r_breadth", "r_gini"]].to_numpy(),
                                Rc.sort_values("child")[["r_count", "r_dur", "r_breadth", "r_gini"]].to_numpy()))
    M = R.merge(c[["child", "group"] + FEATS], on="child", how="outer")
    rows = []
    for a, b, j in zip(["r_count", "r_dur", "r_mdur", "r_breadth", "r_depth", "r_gini"], FEATS, FEATS_JP):
        diff = (M[a] - M[b]).abs()
        rows.append(dict(指標=j, 一致数=int((diff < 1e-6).sum()), 児童数=len(M), 最大差=diff.max()))
    g27 = (M["r_gini27"] - M["view_gini"]).abs()
    rows.append(dict(指標="（参考）参照の集中度：分母27固定で算出した場合", 一致数=int((g27 < 1e-6).sum()), 児童数=len(M), 最大差=g27.max()))
    chk = pd.DataFrame(rows)
    print(chk.to_string(index=False))
    assert np.allclose(M.r_gini, M.r_gini_mad)
    v = views(L)
    allv = L[L.Action == "ew_view"]
    info = dict(n_view_all=int(len(allv)), n_view_teacher=int((~allv.ew_id.astype(str).str.startswith("児童")).sum()),
                n_view_self=int((allv.ew_id == allv.user_id).sum()), n_view_peer=int(len(v)),
                pages=sorted(v.page_number.unique().tolist()), date_min=str(pd.to_datetime(v.date).min().date()),
                date_max=str(pd.to_datetime(v.date).max().date()), dur_max=float(v.duration_sec.max()),
                dur_min=float(v.duration_sec.min()), n_children=int(v.user_id.nunique()),
                breadth_27=int((M.r_breadth == 27).sum()), xlsx_csv_identical=same_src)
    print("ログ情報:", info)
    M.to_csv(OUT / "01_T2_指標再計算_照合.csv", index=False, encoding="utf-8-sig")
    chk.to_csv(OUT / "01_T2_指標照合サマリ.csv", index=False, encoding="utf-8-sig")

    defs = [
        ("量", "閲覧回数", "展開部（p4〜p9）に他の児童のワークシートを開いた操作の回数",
         "V_i = 児童iの閲覧操作（他児童のワークシート）の記録件数"),
        ("時間", "閲覧時間", "展開部の他児童のワークシートの閲覧に費やした総時間（秒）",
         "T_i = Σ（各閲覧操作の継続秒数）"),
        ("時間", "平均閲覧時間/回", "閲覧1回あたりの平均時間（秒）", "T_i / V_i"),
        ("分布", "参照の広さ", "閲覧した異なる児童の数（最大27名）", "|U_i|（U_i：児童iが1回以上閲覧した他児童の集合）"),
        ("分布", "参照の深さ", "同一の他児童に対する平均閲覧回数", "V_i / |U_i|"),
        ("分布", "参照の集中度", "閲覧した他児童（U_i）の間での閲覧回数の偏り（ジニ係数）．0は閲覧した他児童を同じ回数ずつ閲覧，"
         "値が大きいほど特定の児童に閲覧が集中．閲覧しなかった児童は含めない",
         "G_i = Σ_{j∈U_i} Σ_{k∈U_i} |x_ij − x_ik| / (2|U_i|² x̄_i)（x_ij：児童iが児童jを閲覧した回数，x̄_i：U_i内の平均）"),
    ]
    with open(OUT / "01_T2_指標の定義.txt", "w", encoding="utf-8") as f:
        f.write("表2（改）他者参照に関する指標（ジニ係数は実際の算出に合わせた定義）\n\n")
        for d_ in defs:
            f.write(" | ".join(d_) + "\n")
        f.write("\n照合結果（生ログ → clustering_results.csv）\n" + chk.to_string(index=False) + "\n")
        f.write("\nログ情報: " + json.dumps(info, ensure_ascii=False) + "\n")
        f.write("注：閲覧時間 duration_sec はログに付与済みの列をそのまま合計した。閲覧の中断判定（原稿の"
                "『60秒超の操作空白で中断』）はこの列の作成段階の処理であり、本スクリプトでは再現していない。\n")
    S["T2"] = dict(defs=defs, check=chk.to_dict("records"), info=info, R=M)


def sec_T6(c, S):
    H1("[表6] 3群の6指標 平均（SD）（検定は行わない）")
    rows = []
    for g in ORDER:
        s = c[c.group == g]
        rows.append(dict(群=g, 旧名=OLD_NAME[g], n=len(s), **{j: msd(s[f], 2) for f, j in zip(FEATS, FEATS_JP)}))
    s = c
    rows.append(dict(群="全体", 旧名="", n=len(s), **{j: msd(s[f], 2) for f, j in zip(FEATS, FEATS_JP)}))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    T.to_csv(OUT / "02_T6_群別6指標_平均SD.csv", index=False, encoding="utf-8-sig")
    med = []
    for g in ORDER:
        s = c[c.group == g]
        med.append(dict(群=g, **{j: f"{s[f].median():.2f} [{s[f].quantile(.25):.2f}, {s[f].quantile(.75):.2f}]" for f, j in zip(FEATS, FEATS_JP)}))
    pd.DataFrame(med).to_csv(OUT / "02_T6補_群別6指標_中央値IQR.csv", index=False, encoding="utf-8-sig")
    S["T6"] = rows


def sec_T7(c, S, B, fast):
    H1("[表7（新）] クラスタ数の評価と安定性、指標の相互依存、感度分析")
    X = StandardScaler().fit_transform(c[FEATS])
    orig = c.group.map({g: i for i, g in enumerate(ORDER)}).to_numpy()
    rng = np.random.default_rng(SEED + 7)

    # --- k = 2..6
    krows, labels = [], {}
    for k in range(2, 7):
        m = km(X, k, SEED, n_init=200)
        lab = m.labels_
        labels[k] = lab
        krows.append(dict(k=k, WCSS=m.inertia_, silhouette=silhouette_score(X, lab), CH=calinski_harabasz_score(X, lab),
                          DB=davies_bouldin_score(X, lab), sizes="/".join(map(str, sorted(np.bincount(lab), reverse=True)))))
    K = pd.DataFrame(krows)
    ari_k3 = adjusted_rand_score(orig, labels[3])
    print(K.round(3).to_string(index=False))
    print("k=3解と原解（clustering_results.csv）のARI:", round(ari_k3, 4))
    # k=4〜6 で原群がどう分かれるか
    split_desc = {}
    for k in [4, 5, 6]:
        ct = pd.crosstab(c.group, labels[k]).reindex(ORDER)
        print(f"k={k} と原3群のクロス表\n{ct}")
        desc = {}
        for g in ORDER:
            row = ct.loc[g]
            parts = sorted([int(x) for x in row[row > 0]], reverse=True)
            # 原群が単独の1クラスタを構成しているか
            intact = len(parts) == 1 and ct[row[row > 0].index[0]].sum() == row.sum()
            desc[g] = dict(parts=parts, intact=bool(intact))
        split_desc[k] = desc

    # --- Gap 統計量
    Bg = 20 if fast else 200
    mins, maxs = X.min(0), X.max(0)
    refs = [rng.uniform(mins, maxs, size=X.shape) for _ in range(Bg)]

    def logW(Xm, k, seed, n_init):
        if k == 1:
            return np.log(((Xm - Xm.mean(0)) ** 2).sum())
        return np.log(km(Xm, k, seed, n_init).inertia_)
    grows = []
    for k in range(1, 8):
        lw = logW(X, k, SEED, 200)
        lr = np.array([logW(R_, k, SEED + b, 10) for b, R_ in enumerate(refs)])
        grows.append(dict(k=k, gap=lr.mean() - lw, sk=lr.std(ddof=0) * math.sqrt(1 + 1 / Bg)))
    G = pd.DataFrame(grows)
    k_1se = next(int(G.k[i]) for i in range(len(G) - 1) if G.gap[i] >= G.gap[i + 1] - G.sk[i + 1])
    k_gapmax = int(G.loc[G.k.between(2, 6), "gap"].idxmax() + 1)
    print(G.round(3).to_string(index=False))
    print("Gap 1-SE則のk:", k_1se, " Gap最大のk（2〜6）:", k_gapmax)
    K = K.merge(G[["k", "gap", "sk"]], on="k", how="left")
    K.to_csv(OUT / "03_T7_クラスタ数の評価.csv", index=False, encoding="utf-8-sig")
    G.to_csv(OUT / "03_T7補_Gap統計量.csv", index=False, encoding="utf-8-sig")

    # --- ブートストラップ（復元抽出）と80%部分標本の群別Jaccard
    def resample_jaccard(mode, Bn):
        J = {g: [] for g in range(3)}
        aris = []
        n = len(X)
        for b in range(Bn):
            idx = rng.integers(0, n, n) if mode == "boot" else np.sort(rng.choice(n, int(round(0.8 * n)), replace=False))
            mb = km(X[idx], 3, int(rng.integers(0, 2 ** 31 - 1)), n_init=20)
            uniq = np.unique(idx)
            lab_u = mb.predict(X[uniq])
            for g in range(3):
                A = set(uniq[orig[uniq] == g])
                if not A:
                    continue
                J[g].append(max(len(A & set(uniq[lab_u == kk])) / len(A | set(uniq[lab_u == kk])) for kk in range(3)))
            aris.append(adjusted_rand_score(orig, mb.predict(X)))
        return J, np.array(aris)
    Jb, ARb = resample_jaccard("boot", B)
    Js, ARs = resample_jaccard("sub", B)
    jrows = []
    for gi, g in enumerate(ORDER):
        a, s_ = np.array(Jb[gi]), np.array(Js[gi])
        jrows.append(dict(群=g, n=int((orig == gi).sum()), boot_mean=a.mean(), boot_median=np.median(a), boot_lt05=(a < .5).mean(),
                          boot_ge075=(a >= .75).mean(), boot_valid=len(a), sub_mean=s_.mean(), sub_lt05=(s_ < .5).mean(), sub_valid=len(s_)))
    JT = pd.DataFrame(jrows)
    print(JT.round(3).to_string(index=False))
    stab = dict(boot_ari_mean=ARb.mean(), boot_ari_eq1=np.isclose(ARb, 1).mean(), sub_ari_mean=ARs.mean(),
                sub_ari_eq1=np.isclose(ARs, 1).mean(), B=B, n_sub=int(round(0.8 * 28)))
    print("再標本の全28名割当てと原解のARI:", {k: round(v, 3) for k, v in stab.items()})
    JT.to_csv(OUT / "03_T7_安定性_Jaccard.csv", index=False, encoding="utf-8-sig")

    # --- 80%部分標本でシルエット最大となった k
    best = []
    for b in range(B):
        idx = rng.choice(28, 22, replace=False)
        Xs = X[idx]
        sil = [silhouette_score(Xs, km(Xs, k, int(rng.integers(0, 2 ** 31 - 1)), n_init=10).labels_) for k in range(2, 7)]
        best.append(2 + int(np.argmax(sil)))
    bestk = pd.Series(best).value_counts(normalize=True).sort_index()
    print("80%部分標本でシルエット最大となったkの割合:", bestk.round(3).to_dict())
    bestk.rename("割合").to_csv(OUT / "03_T7補_部分標本のシルエット最大k.csv", encoding="utf-8-sig")

    # --- 個体シルエット
    ss = silhouette_samples(X, orig)
    sil_ind = pd.DataFrame(dict(child=c.child, group=c.group, silhouette=ss))
    sil_ind.to_csv(OUT / "03_T7補_個体シルエット.csv", index=False, encoding="utf-8-sig")
    low = sil_ind.sort_values("silhouette").head(3)

    # --- 指標間の順位相関とVIF
    rho = c[FEATS].corr(method="spearman")
    rho.index = rho.columns = FEATS_JP
    rho.to_csv(OUT / "03_T7補_6指標Spearman相関.csv", encoding="utf-8-sig")
    off = rho.where(~np.eye(6, dtype=bool)).abs().stack()
    pair = off.idxmax()
    Xc = np.column_stack([np.ones(28), X])
    vif = {j: variance_inflation_factor(Xc, i + 1) for i, j in enumerate(FEATS_JP)}
    print(rho.round(3).to_string())
    print("最大の|ρ|:", pair, round(rho.loc[pair], 3))
    print("VIF:", {k: round(v, 1) for k, v in vif.items()})
    pd.Series(vif, name="VIF").to_csv(OUT / "03_T7補_VIF.csv", encoding="utf-8-sig")

    # --- 感度分析（k=3解と原解のARI）
    sens = []

    def add(name, lab, note="", sil=None):
        mapped = map_labels(orig, lab, 3)
        sens.append(dict(分析=name, ARI=adjusted_rand_score(orig, lab), 所属が変わった児童数=int((mapped != orig).sum()),
                         群の大きさ="/".join(map(str, sorted(np.bincount(lab), reverse=True))), シルエットk3=sil, 備考=note))
    F4 = ["view_count", "view_duration", "view_breadth", "view_gini"]
    X4 = StandardScaler().fit_transform(c[F4])
    sil4 = {k: silhouette_score(X4, km(X4, k, SEED).labels_) for k in range(2, 7)}
    add("導出2指標（平均閲覧時間/回・参照の深さ）を除いた4指標", km(X4, 3, SEED).labels_,
        f"シルエット最大k={max(sil4, key=sil4.get)}（k=3 {sil4[3]:.3f}、k=4 {sil4[4]:.3f}）", sil4[3])
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    nc80 = int(np.argmax(cum >= .80) + 1)
    Z = PCA(n_components=nc80).fit_transform(X)
    add(f"主成分得点（累積寄与率80%以上：{nc80}成分，{cum[nc80 - 1] * 100:.1f}%）", km(Z, 3, SEED).labels_, "白色化なし",
        silhouette_score(Z, km(Z, 3, SEED).labels_))
    nc90 = int(np.argmax(cum >= .90) + 1)
    Z3 = PCA(n_components=nc90).fit_transform(X)
    add(f"（参考）主成分得点（累積寄与率90%以上：{nc90}成分，{cum[nc90 - 1] * 100:.1f}%）", km(Z3, 3, SEED).labels_, "白色化なし",
        silhouette_score(Z3, km(Z3, 3, SEED).labels_))
    add("Ward法（6指標）", AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X))
    g27 = S["T2"]["R"].set_index("child")["r_gini27"]
    c27 = c.copy()
    c27["view_gini"] = c27.child.map(g27)
    X27 = StandardScaler().fit_transform(c27[FEATS])
    lab27 = km(X27, 3, SEED).labels_
    sil27 = {k: silhouette_score(X27, km(X27, k, SEED).labels_) for k in range(2, 7)}
    add("参照の集中度を分母27固定で算出した6指標", lab27, f"シルエット最大k={max(sil27, key=sil27.get)}（k=3 {sil27[3]:.3f}）", sil27[3])
    # 補足（本文には出さない）
    add("（補足）群平均法（6指標）", AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(X))
    add("（補足）完全連結法（6指標）", AgglomerativeClustering(n_clusters=3, linkage="complete").fit_predict(X))
    add("（補足）混合正規分布・球状共分散", GaussianMixture(3, covariance_type="spherical", n_init=20, random_state=SEED).fit_predict(X))
    add("（補足）混合正規分布・対角共分散", GaussianMixture(3, covariance_type="diag", n_init=20, random_state=SEED).fit_predict(X))
    Dm = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    bestmed = min(itertools.combinations(range(28), 3), key=lambda med: Dm[:, med].min(1).sum())
    add("（補足）k-medoids（PAM・全探索）", Dm[:, bestmed].argmin(1))
    Xlog = StandardScaler().fit_transform(np.column_stack([np.log(c[f]) if f != "view_gini" else c[f] for f in FEATS]))
    add("（補足）6指標の対数変換", km(Xlog, 3, SEED).labels_)
    F5 = ["view_count", "view_duration", "view_mean_duration", "view_breadth", "view_depth"]
    add("（補足）参照の集中度を除いた5指標", km(StandardScaler().fit_transform(c[F5]), 3, SEED).labels_)
    SE = pd.DataFrame(sens)
    print(SE.round(3).to_string(index=False))
    SE.to_csv(OUT / "03_T7_感度分析_ARI.csv", index=False, encoding="utf-8-sig")
    print("PCA 累積寄与率:", np.round(cum, 3))

    S["T7"] = dict(K=K, G=G, k_1se=k_1se, k_gapmax=k_gapmax, ari_k3=ari_k3, split=split_desc, J=JT, stab=stab, bestk=bestk,
                   rho=rho, pair=pair, rho_max=float(rho.loc[pair]), vif=vif, sens=SE, cum=cum, nc80=nc80, low_sil=low,
                   labels4=labels[4], X=X, orig=orig, Bg=Bg)


def sec_T8(d, S):
    H1("[表8] LLM評価の信頼性指標（既存値の再現）と人手再評定の母集団・抽出率")
    s50 = pd.read_csv(ROOT / "03_LLM評価データ/振り返り記述_50サンプル_層化抽出.csv", encoding="utf-8-sig")
    s50 = s50.rename(columns={"児童ID": "child", "ページ": "page"})
    m = s50.merge(d[["child", "page", "total"]], on=["child", "page"], how="left")
    assert m.total.notna().all()
    h = m["総合判定"].map(R4).to_numpy()
    l = m["total"].to_numpy()
    res = dict(
        kappa_w=cohen_kappa_score(h, l, weights="linear"),
        kappa_bin=cohen_kappa_score(h >= 3, l >= 3),
        F1=f1_score(h >= 3, l >= 3),
        exact=(h == l).mean(), adjacent=(np.abs(h - l) <= 1).mean(), bin_agree=((h >= 3) == (l >= 3)).mean(), n=len(m))
    Mx = np.column_stack([d[f"{p}_総合判定"].map(R4) for p in ["OpenAI", "Anthropic", "Google"]]).astype(float)
    res["ICC2k"] = icc2k(Mx)
    cr = np.corrcoef(Mx.T)
    res["r_pairs"] = {"OpenAI–Anthropic": cr[0, 1], "OpenAI–Google": cr[0, 2], "Anthropic–Google": cr[1, 2]}
    # 母集団：ルーブリック148記述のうち既存Claude SOLOがある記述
    sc = load_solo_claude()
    pop = d[["child", "page"]].merge(sc, on=["child", "page"], how="inner")
    in_pop = s50.merge(pop, on=["child", "page"], how="left").solo.notna().all()
    strata = pd.DataFrame(dict(母集団=pop.solo.value_counts().sort_index(),
                               抽出=s50["既存SOLO_Claude"].value_counts().sort_index())).fillna(0).astype(int)
    strata["抽出率"] = (strata["抽出"] / strata["母集団"]).round(3)
    res.update(pop_n=len(pop), rate=len(s50) / len(pop), all_in_pop=bool(in_pop), claude_rows=len(sc),
               claude_not_in_rubric=sc.merge(d[["child", "page"]], how="left", indicator=True).query("_merge=='left_only'")[["child", "page"]].to_dict("records"),
               rubric_without_claude=d[["child", "page"]].merge(sc, how="left", indicator=True).query("_merge=='left_only'")[["child", "page"]].to_dict("records"))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    print(strata)
    rows = [("Cohen's κ<sub>w</sub>（4段階・線形重み付け，主指標）", fz(res["kappa_w"])),
            ("Cohen's κ（A以上 vs B以下の二値化，補助指標）", fz(res["kappa_bin"])),
            ("F1（A以上の検出）", fz(res["F1"])),
            ("完全一致率", f"{res['exact'] * 100:.1f}%"),
            ("隣接一致率（差≦1）", f"{res['adjacent'] * 100:.1f}%"),
            ("3LLM相互の評価一致度 ICC(2,3)", fz(res["ICC2k"])),
            ("3LLM間 Pearson相関 r", f"{fz(min(res['r_pairs'].values()), 2)}〜{fz(max(res['r_pairs'].values()), 2)}")]
    pd.DataFrame(rows, columns=["指標", "値"]).to_csv(OUT / "04_T8_信頼性指標.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(OUT / "04_T8補_人手再評定の層別抽出.csv", encoding="utf-8-sig")
    S["T8"] = dict(res=res, rows=rows, strata=strata)


def sec_T9(child_total, S):
    H1("[表9] 児童ごとの平均段階の分布と最終振り返り段階の分布")
    a = child_total
    mean_stage = pd.cut(a["mean"], [0, 1.5, 2.5, 3.5, 5], labels=["C", "B", "A", "S"], right=False)
    last_stage = a["last"].map(INV4)
    p9_stage = a["p9"].dropna().map(INV4)

    def dist(s):
        n = len(s)
        return {k: (int((s == k).sum()), (s == k).mean() * 100) for k in ["S", "A", "B", "C"]}, n
    dm, nm = dist(mean_stage.astype(str))
    dl, nl = dist(last_stage)
    d9, n9 = dist(p9_stage)
    print("平均段階:", dm, nm)
    print("最終振り返り段階:", dl, nl)
    print("p9のみ:", d9, n9)
    fpcount = a.last_page.value_counts().sort_index().to_dict()
    print("最終振り返りのページ:", fpcount)
    rows = []
    for nm_, dd, n in [("児童ごとの平均段階", dm, nm), ("最終振り返りの段階", dl, nl), ("（参考）p9（第6場面）の段階", d9, n9)]:
        rows.append(dict(区分=nm_, **{k: f"{dd[k][0]} ({dd[k][1]:.1f})" for k in ["S", "A", "B", "C"]}, 計=f"{n} (100.0)"))
    pd.DataFrame(rows).to_csv(OUT / "05_T9_深まりの分布.csv", index=False, encoding="utf-8-sig")
    S["T9"] = dict(dm=dm, dl=dl, d9=d9, n9=n9, rows=rows, fpcount=fpcount,
                   n_last_A=int((a["last"] >= 3).sum()), n_p9_A=int((a["p9"] >= 3).sum()), n_p9=int(a["p9"].notna().sum()),
                   n_any=int(a["ach_any"].sum()))


def sec_T10(pg, S, solo_label):
    H1(f"[表10] ルーブリック × {solo_label}")
    m = pg.dropna(subset=["solo"]).copy()
    m["solo"] = m.solo.astype(int)
    m["rub"] = m.total.map(INV4)
    ct = pd.crosstab(m.rub, m.solo).reindex(index=["C", "B", "A", "S"], columns=[1, 2, 3, 4, 5]).fillna(0).astype(int)
    rho, p = stats.spearmanr(m.total, m.solo)
    tau = stats.kendalltau(m.total, m.solo).statistic
    rubA = (m.total >= 3).astype(int)
    soloR = (m.solo >= 4).astype(int)
    agree = (rubA == soloR).mean()
    kap = cohen_kappa_score(rubA, soloR)
    within = []
    for r in ["C", "B", "A", "S"]:
        s = m.loc[m.rub == r, "solo"]
        within.append(dict(段階=r, n=len(s), SOLO平均=s.mean(), SD=s.std(ddof=1), 最小=s.min(), 最大=s.max(), 段階数=s.nunique(),
                           最頻値=int(s.mode().iloc[0]), 最頻値以外の割合=(s != s.mode().iloc[0]).mean()))
    W = pd.DataFrame(within)
    t1 = m[(m.total <= 2) & (m.solo >= 4)]
    t2 = m[(m.total >= 3) & (m.solo <= 3)]

    def prho(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

        def res_(a, b):
            b1 = np.column_stack([np.ones(len(b)), b])
            return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
        return np.corrcoef(res_(rx, rz), res_(ry, rz))[0, 1]
    el = {e: stats.spearmanr(m.solo, m[e])[0] for e in ["eA", "eB", "eC"]}
    out = dict(n=len(m), rho=rho, p=p, rho2=rho ** 2, tau=tau, agree=agree, kappa=kap, n_agree=int((rubA == soloR).sum()),
               t1=len(t1), t2=len(t2), t1_nchar_med=t1.nchar.median(), t2_nchar_med=t2.nchar.median(),
               rho_solo_nchar=stats.spearmanr(m.solo, m.nchar)[0], rho_rub_nchar=stats.spearmanr(m.total, m.nchar)[0],
               prho=prho(m.total, m.solo, m.nchar), el=el,
               short_solo4=int(((m.nchar <= m.nchar.quantile(.25)) & (m.solo >= 4)).sum()), q25=m.nchar.quantile(.25),
               solo4_min_nchar=int(m.loc[m.solo >= 4, "nchar"].min()) if (m.solo >= 4).any() else np.nan)
    print(ct)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    print(W.round(2).to_string(index=False))
    ct2 = ct.copy()
    ct2["計"] = ct2.sum(1)
    ct2.loc["計"] = ct2.sum(0)
    ct2.to_csv(OUT / "06_T10_ルーブリックxSOLO_クロス表.csv", encoding="utf-8-sig")
    W.to_csv(OUT / "06_T10補_段階内SOLO分布.csv", index=False, encoding="utf-8-sig")
    cases = pd.concat([t1.assign(型="(1) SOLO関係構造以上・ルーブリックB以下"), t2.assign(型="(2) ルーブリックA以上・SOLO多重構造以下")])
    cases = cases[["型", "child", "page", "rub", "eA", "eB", "eC", "solo", "nchar", "text"]].copy()
    cases[["eA", "eB", "eC"]] = cases[["eA", "eB", "eC"]].apply(lambda s: s.map(INV3))
    cases["text"] = cases["text"].map(mask_names)
    cases.to_csv(OUT / "06_T10補_不一致2型の記述一覧.csv", index=False, encoding="utf-8-sig")
    S["T10"] = dict(ct=ct2, out=out, W=W)


def sec_T11(child_total, pg, c, S, B, rng):
    H1("[表11（新）] 3群別の深まりスコア（原総合）と群内の前半→後半、KW、変化量の群間比較")
    a = child_total
    pats = [("mean", "平均"), ("last", "最終振り返り"), ("max", "最大"), ("front", "前半"), ("back", "後半"), ("change", "変化量")]
    rows, wil = [], {}
    for g in ORDER + ["全体"]:
        s = a if g == "全体" else a[a.group == g]
        w = wilcoxon_exact(s.change)
        wil[g] = w
        rows.append(dict(群=g, n=len(s), 平均=msd(s["mean"]), 最終振り返り=msd(s["last"]), 最大=msd(s["max"]), 前半=msd(s["front"]),
                         後半=msd(s["back"]), 変化量Δ=msd_sign(s["change"]), W=w["W"], 正確p=w["p_exact"], r=w["r"], r_rb=w["r_rb"],
                         非ゼロ差=w["n_nonzero"], 最小可能p=w["p_min"], 増加=w["n_pos"], 減少=w["n_neg"]))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    T.to_csv(OUT / "07_T11_群別記述統計と群内検定.csv", index=False, encoding="utf-8-sig")
    kw = {}
    krows = []
    for col, lab in pats:
        r = kw_test(a, col)
        r["p_perm"] = kw_perm(a, col, rng, B=B * 20)
        kw[col] = r
        krows.append(dict(パターン=lab, H=r["H"], df=2, p=r["p"], p_順列=r["p_perm"], epsilon2=r["eps2"], eta2H_旧式=r["eta2H"]))
    KT = pd.DataFrame(krows)
    print(KT.round(4).to_string(index=False))
    KT.to_csv(OUT / "07_T11_KW_6パターン.csv", index=False, encoding="utf-8-sig")

    # 変化量の群間比較の補強
    chg = {g: a.loc[a.group == g, "change"].to_numpy() for g in ORDER}
    boot = {}
    for g1, g2 in [("広範・中頻度群", "広範・高頻度群"), ("広範・中頻度群", "限定・長時間群"), ("限定・長時間群", "広範・高頻度群")]:
        bs = [rng.choice(chg[g1], len(chg[g1])).mean() - rng.choice(chg[g2], len(chg[g2])).mean() for _ in range(B * 10)]
        boot[(g1, g2)] = (chg[g1].mean() - chg[g2].mean(), *np.percentile(bs, [2.5, 97.5]))
    dd = pg.merge(c[["child", "group"]], on="child").copy()
    dd["half"] = (dd.page >= 7).astype(int)
    dd["y"] = dd.total
    m1 = smf.mixedlm("y ~ C(group, Treatment('広範・中頻度群')) * half", dd, groups=dd["child"]).fit(reml=False)
    m0 = smf.mixedlm("y ~ C(group, Treatment('広範・中頻度群')) + half", dd, groups=dd["child"]).fit(reml=False)
    lrt = 2 * (m1.llf - m0.llf)
    lmm = dict(chi2=lrt, df=2, p=stats.chi2.sf(lrt, 2))
    # 検出力（ANOVA近似と再標本化KW）
    ns = [len(chg[g]) for g in ORDER]
    mus = [chg[g].mean() for g in ORDER]
    sds = [chg[g].std(ddof=1) for g in ORDER]
    N = sum(ns)
    grand = sum(n * mu for n, mu in zip(ns, mus)) / N
    sp = math.sqrt(sum((n - 1) * s ** 2 for n, s in zip(ns, sds)) / (N - 3))
    f = math.sqrt(sum(n * (mu - grand) ** 2 for n, mu in zip(ns, mus)) / N) / sp
    crit = stats.f.ppf(.95, 2, N - 3)
    power = stats.ncf.sf(crit, 2, N - 3, f ** 2 * N)
    f80 = next(ff for ff in np.arange(0.05, 2, 0.01) if stats.ncf.sf(crit, 2, N - 3, ff ** 2 * N) >= .8)
    N80 = next(NN for NN in range(28, 5000) if stats.ncf.sf(stats.f.ppf(.95, 2, NN - 3), 2, NN - 3, f ** 2 * NN) >= .8)
    hits = 0
    nsim = B * 5
    for _ in range(nsim):
        gs = [rng.choice(chg[g], n) for g, n in zip(ORDER, ns)]
        try:
            hits += stats.kruskal(*gs)[1] < .05
        except ValueError:
            pass
    pw = dict(f=f, power_anova=power, f80=f80, N80=N80, power_kw=hits / nsim)
    print("変化量の差（ブートストラップ95%区間）:", {f"{k[0]}−{k[1]}": tuple(round(x, 3) for x in v) for k, v in boot.items()})
    print("LMM 群×時期 交互作用:", {k: round(v, 3) for k, v in lmm.items()})
    print("検出力:", {k: round(v, 3) for k, v in pw.items()})
    with open(OUT / "07_T11補_変化量の群間比較.txt", "w", encoding="utf-8") as fh:
        fh.write(f"変化量のKW：H(2) = {kw['change']['H']:.3f}, p = {kw['change']['p']:.4f}, 順列p = {kw['change']['p_perm']:.4f}, ε² = {kw['change']['eps2']:.3f}\n")
        for k, v in boot.items():
            fh.write(f"変化量の差 {k[0]}−{k[1]}：{v[0]:+.3f} [95%CI {v[1]:+.3f}, {v[2]:+.3f}]（ブートストラップ{B * 10}回）\n")
        fh.write(f"線形混合モデル（記述単位148、児童ランダム切片、最尤法）群×時期（前半/後半）の交互作用 LRT：χ²(2) = {lrt:.3f}, p = {lmm['p']:.4f}\n")
        fh.write(f"検出力：観測 Cohen's f = {f:.3f} → ANOVA近似の検出力 {power:.3f}；再標本化KWの検出力 {pw['power_kw']:.3f}（{nsim}回）；"
                 f"n = 6/18/4 で80%に必要な f ≈ {f80:.2f}；観測fで80%に必要な総N ≈ {N80}\n")
        for g in ORDER:
            fh.write(f"群内正確検定の最小可能p（{g}、非ゼロ差{wil[g]['n_nonzero']}名）：{wil[g]['p_min']:.4f}\n")
    S["T11"] = dict(T=T, wil=wil, kw=kw, boot=boot, lmm=lmm, pw=pw, a=a)


def sec_T12(child_total, S):
    H1("[表12（新）] 単元目標達成の5定義と群別達成率（Fisher–Freeman–Halton正確検定）")
    a = child_total
    defs = [("ach_major", "① 平均ベース（A以上の記述が半数以上）"), ("ach_last", "② 最終振り返りでA以上"),
            ("ach_any", "③ 1回以上A以上"), ("ach_front", "④ 前半（p4〜p6）でA以上が半数以上"),
            ("ach_back", "⑤ 後半（p7〜p9）でA以上が半数以上"), ("ach_p9", "（感度分析）② p9のみ（*n* = 19）")]
    rows, res = [], {}
    for col, lab in defs:
        r = crosstab_test(a, col)
        res[col] = r
        cnts = r["counts"]
        tot = (sum(x[0] for x in cnts), sum(x[1] for x in cnts))
        rows.append(dict(定義=lab, **{g: frac(*cnts[i]) for i, g in enumerate(ORDER)}, 計=frac(*tot), FFH_p=r["p_ffh"],
                         CramerV=r["V"], chi2=r["chi2"], 最小期待度数=r["min_exp"],
                         中頻度群が最高=bool(r["rates"][1] > max(r["rates"][0], r["rates"][2]))))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    T.to_csv(OUT / "08_T12_達成5定義.csv", index=False, encoding="utf-8-sig")
    fp_by_group = pd.crosstab(a.group, a.last_page).reindex(ORDER)
    print("最終振り返りのページ（群別）:\n", fp_by_group)
    fp_by_group.to_csv(OUT / "08_T12補_最終振り返りのページ_群別.csv", encoding="utf-8-sig")
    S["T12"] = dict(T=T, res=res, fp_by_group=fp_by_group)


def sec_T432(pg, c, final_page, S, solo_label):
    H1("[4.3.2 表] 他者参照の要素を含まない評価による確認（判定基準は事前に固定）")
    inds = [("total", 3, "原総合（要素1〜3）", False), ("AB_rule", 3, "要素1・2のみ（規則版）", False),
            ("eC", 3, "要素3のみ（参考）", False), ("solo", 4, f"{solo_label}：Level 4以上", False),
            ("solo", 4, f"（感度）{solo_label}：最終振り返りが欠損の児童は直前の記述で代替", True),
            ("AB_mean", 3, "（補足）要素1・2のみ（平均版：両要素A）", False)]
    rows, keep = [], {}
    for col, thr, lab, alt in inds:
        a = aggregate(pg, col, thr, final_page, alt_last=alt).merge(c[["child", "group"]], on="child")
        k1 = kw_test(a, "mean")
        r2 = crosstab_test(a, "ach_last")
        r9 = crosstab_test(a, "ach_p9")
        w = {g: wilcoxon_exact(a.loc[a.group == g, "change"].dropna()) for g in ORDER}
        wa = wilcoxon_exact(a.change.dropna())
        k3 = kw_test(a.dropna(subset=["change"]), "change")  # 変化量の群間差（再査読①M2で追加）
        keep[lab] = dict(a=a, k1=k1, r2=r2, r9=r9, w=w, wa=wa, k3=k3, col=col, thr=thr)
        rows.append({
            "指標": lab, "記述数": int(pg[col].notna().sum()), "①平均KW_p": k1["p"], "①H": k1["H"],
            "②最終_到達率": "／".join(frac(*x, d=0) for x in r2["counts"]), "②n": r2["n"], "②FFH_p": r2["p_ffh"], "②V": r2["V"],
            "②p9_到達率": "／".join(frac(*x, d=0) for x in r9["counts"]), "②p9_n": r9["n"], "②p9_FFH_p": r9["p_ffh"], "②p9_V": r9["V"],
            **{f"③Δ_{g}": w[g]["delta_M"] for g in ORDER}, **{f"③p_{g}": w[g]["p_exact"] for g in ORDER},
            "③変化量KW_H": k3["H"], "③変化量KW_p": k3["p"], "③変化量KW_eps2": k3["eps2"],
            "学級全体Δ": wa["delta_M"], "学級全体W": wa["W"], "学級全体p": wa["p_exact"], "学級全体r": wa["r"]})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    T.to_csv(OUT / "09_T432_他者参照を含まない評価.csv", index=False, encoding="utf-8-sig")

    def cond(lab):
        r = keep[lab]["r2"]
        rates = r["rates"]
        mid_high = bool(rates[1] > max(rates[0], rates[2]))
        vok = (not isnan(r["V"])) and r["V"] >= .30
        return dict(mid_highest=mid_high, V=r["V"], V_ok=vok, met=mid_high and vok, rates=rates)

    def decide(labA, labB):
        cA, cB = cond(labA), cond(labB)
        if cA["met"] and cB["met"]:
            sc = "S1（維持）"
        elif cA["mid_highest"] or cB["mid_highest"]:
            sc = "S2（部分的維持）"
        else:
            sc = "S3（非維持）"
        return sc, cA, cB
    labAB = "要素1・2のみ（規則版）"
    labS = f"{solo_label}：Level 4以上"
    labS2 = f"（感度）{solo_label}：最終振り返りが欠損の児童は直前の記述で代替"
    sc, cA, cB = decide(labAB, labS)
    sc2, _, cB2 = decide(labAB, labS2)
    txt = (f"判定基準（分析前に固定、修正方針§5.2）：S1＝要素1・2判定と中立SOLOの両方で、最終振り返りの到達率が広範・中頻度群で最高"
           f"かつCramér's V ≥ .30／S2＝一方のみ成立、または方向は一致するがV < .30／S3＝両指標で広範・中頻度群最高が成り立たない。p値は判定に用いない。\n"
           f"要素1・2：到達率 {[round(x, 3) for x in cA['rates']]}，中頻度最高={cA['mid_highest']}，V={cA['V']:.3f} → 成立={cA['met']}\n"
           f"SOLO（{solo_label}）：到達率 {[round(x, 3) if not isnan(x) else None for x in cB['rates']]}，中頻度最高={cB['mid_highest']}，"
           f"V={cB['V'] if isnan(cB['V']) else round(cB['V'], 3)} → 成立={cB['met']}\n"
           f"判定：{sc}\n（感度：SOLOの欠損を直前記述で代替した場合の判定：{sc2}；SOLO到達率 {[round(x, 3) for x in cB2['rates']]}, V={cB2['V']:.3f}）\n"
           f"注：SOLOが既存Claude評価の場合、中立基準ではないため判定は暫定。3LLM再評価後に再実行して確定する。\n")
    print(txt)
    with open(OUT / "09_T432_判定.txt", "w", encoding="utf-8") as fh:
        fh.write(txt)
    S["T432"] = dict(T=T, keep=keep, scenario=sc, scenario_alt=sc2, cA=cA, cB=cB, cB2=cB2, labAB=labAB, labS=labS, labS2=labS2)


def sec_T433(pg, child_total, raw, c, S):
    H1("[4.3.3] 最終振り返りの記述の特徴（要素別判定）と引用候補一覧")
    a = child_total.merge(c[["child", "group"]].rename(columns={"group": "g2"}), on="child")
    fin = pg.merge(a[["child", "last_page"]], left_on=["child", "page"], right_on=["child", "last_page"])
    fin = fin.merge(c[["child", "group"]], on="child")
    fin["achieved"] = (fin.total >= 3).astype(int)
    rows = []
    for ach in [1, 0]:
        for g in ORDER + ["計"]:
            s = fin[fin.achieved == ach] if g == "計" else fin[(fin.achieved == ach) & (fin.group == g)]
            row = dict(到達=("A以上（到達）" if ach else "B以下（未到達）"), 群=g, n=len(s))
            for e, ej in [("eA", "要素1"), ("eB", "要素2"), ("eC", "要素3")]:
                row[ej] = "/".join(str(int((s[e] == v).sum())) for v in [3, 2, 1])
            rows.append(row)
    T = pd.DataFrame(rows)
    print("（各要素の表記は A/B/C の人数）")
    print(T.to_string(index=False))
    T.to_csv(OUT / "10_T433_最終振り返りの要素別判定_群別.csv", index=False, encoding="utf-8-sig")

    # 代表的な判定理由：9回（3LLM×3回）の要素判定が3LLM中央値の要素判定と最も一致する試行
    rr = raw.dropna(subset=["要素A_本文根拠", "要素B_心情読み取り", "要素C_解釈統合"]).copy()
    prov_order = {"OpenAI": 0, "Anthropic": 1, "Google": 2}
    reps = []
    for _, f in fin.iterrows():
        cand = rr[(rr.child == f.child) & (rr.page == f.page)].copy()
        tgt = (INV3[f.eA], INV3[f.eB], INV3[f.eC])
        cand["match"] = (cand["要素A_本文根拠"] == tgt[0]).astype(int) + (cand["要素B_心情読み取り"] == tgt[1]).astype(int) + (cand["要素C_解釈統合"] == tgt[2]).astype(int)
        cand["tot_match"] = (cand["総合判定"] == INV4[f.total]).astype(int)
        cand["po"] = cand.Provider.map(prov_order)
        cand = cand.sort_values(["match", "tot_match", "po", "試行回"], ascending=[False, False, True, True])
        top = cand.iloc[0] if len(cand) else None
        reps.append(dict(child=f.child, rationale=mask_names(top["判定理由"]) if top is not None else "",
                         rationale_src=f"{top['Provider']}#{int(top['試行回'])}" if top is not None else "",
                         rationale_match=int(top["match"]) if top is not None else np.nan, n_trials=len(cand)))
    R = pd.DataFrame(reps)
    L = fin.merge(R, on="child")
    L["要素1"] = L.eA.map(INV3)
    L["要素2"] = L.eB.map(INV3)
    L["要素3"] = L.eC.map(INV3)
    L["総合"] = L.total.map(INV4)
    L["場面"] = L.page.map(PAGE_SCENE)
    L["記述（姓は〇〇に置換）"] = L.text.map(mask_names)
    L["旧群名"] = L.group.map(OLD_NAME)
    L["到達"] = L.achieved.map({1: "到達", 0: "未到達"})
    L["SOLO"] = L.solo
    cols = ["group", "旧群名", "到達", "child", "page", "場面", "総合", "要素1", "要素2", "要素3", "SOLO", "nchar",
            "記述（姓は〇〇に置換）", "rationale", "rationale_src", "rationale_match"]
    L = L.sort_values(["group", "achieved", "child"], ascending=[True, False, True])[cols]
    L = L.rename(columns={"group": "群", "child": "児童", "page": "ページ", "nchar": "文字数", "rationale": "代表的な判定理由（3LLM×3回から1つ）",
                          "rationale_src": "判定理由の出所", "rationale_match": "要素判定の一致数(0-3)"})
    L.to_csv(OUT / "10_T433_最終振り返り一覧_引用候補.csv", index=False, encoding="utf-8-sig")
    ach = fin[fin.achieved == 1]
    S["T433"] = dict(T=T, n_ach=len(ach), eA_all_A=int((ach.eA == 3).sum()), eB_all_A=int((ach.eB == 3).sum()),
                     eC_A=int((ach.eC == 3).sum()), mid_ach=int(((ach.group == "広範・中頻度群")).sum()),
                     mid_ach_eC_A=int(((ach.group == "広範・中頻度群") & (ach.eC == 3)).sum()), fin=fin)


def sec_length(pg, child_total, child_solo, c, S, B, rng):
    H1("[記述量の交絡] 最終振り返りの文字数")
    fin = S["T433"]["fin"].copy()
    rows = []
    for g in ORDER + ["全体"]:
        s = fin if g == "全体" else fin[fin.group == g]
        rows.append(dict(群=g, n=len(s), 文字数_MSD=msd(s.nchar, 1), 中央値=s.nchar.median()))
    kwn = kw_test(fin.rename(columns={"nchar": "nn"}), "nn")
    ach_n = fin.groupby("achieved").nchar.agg(["mean", "std", "median", "count"])
    m = pg.dropna(subset=["solo"])
    cor = dict(rho_nchar_total_rec=stats.spearmanr(pg.nchar, pg.total)[0], n_rec=len(pg),
               rho_nchar_solo_rec=stats.spearmanr(m.nchar, m.solo)[0], n_solo=len(m),
               rho_nchar_total_final=stats.spearmanr(fin.nchar, fin.total)[0])
    fs_ = fin.dropna(subset=["solo"])
    cor["rho_nchar_solo_final"] = stats.spearmanr(fs_.nchar, fs_.solo)[0]
    # 文字数の中央値で層化した群別到達率
    med = fin.nchar.median()
    fin["stratum"] = np.where(fin.nchar > med, "長い（中央値超）", "短い（中央値以下）")
    strat = []
    for st in ["短い（中央値以下）", "長い（中央値超）"]:
        s = fin[fin.stratum == st]
        strat.append(dict(層=st, n=len(s), **{g: frac(int(s[(s.group == g)].achieved.sum()), int((s.group == g).sum()), 0) for g in ORDER}))
    ST = pd.DataFrame(strat)

    # 層内で群ラベルを並べ替える層化順列検定（統計量＝層別χ²の和）
    def chi_sum(df, labs):
        tot = 0.0
        for st in df.stratum.unique():
            mask = (df.stratum == st).to_numpy()
            t = np.array([[((labs[mask] == g) & (df.achieved.to_numpy()[mask] == 1)).sum(),
                           ((labs[mask] == g) & (df.achieved.to_numpy()[mask] == 0)).sum()] for g in ORDER])
            t = t[t.sum(1) > 0]
            if t.shape[0] < 2 or (t.sum(0) == 0).any():
                continue
            tot += stats.chi2_contingency(t, correction=False)[0]
        return tot
    labs = fin.group.to_numpy()
    obs = chi_sum(fin, labs)
    cnt = 0
    nperm = B * 20
    strata_idx = {st: np.where(fin.stratum.to_numpy() == st)[0] for st in fin.stratum.unique()}
    for _ in range(nperm):
        pl = labs.copy()
        for st, idx in strata_idx.items():
            pl[idx] = rng.permutation(labs[idx])
        cnt += chi_sum(fin, pl) >= obs - 1e-12
    p_strat = (cnt + 1) / (nperm + 1)
    # ロジスティック回帰（中頻度群ダミー＋文字数z）と順序ロジット（最終振り返り段階）
    fin["mid"] = (fin.group == "広範・中頻度群").astype(int)
    fin["z"] = (fin.nchar - fin.nchar.mean()) / fin.nchar.std(ddof=1)
    lr = {}
    try:
        r0 = sm.Logit(fin.achieved, sm.add_constant(fin[["mid"]])).fit(disp=0)
        r1 = sm.Logit(fin.achieved, sm.add_constant(fin[["mid", "z"]])).fit(disp=0)
        lr = dict(b_mid_unadj=r0.params["mid"], p_mid_unadj=r0.pvalues["mid"], b_mid=r1.params["mid"], p_mid=r1.pvalues["mid"],
                  b_z=r1.params["z"], p_z=r1.pvalues["z"], OR_mid=math.exp(r1.params["mid"]))
    except Exception as e:
        lr = dict(error=str(e))
    om = {}
    try:
        o0 = OrderedModel(fin.total.astype(int), fin[["mid"]], distr="logit").fit(method="bfgs", disp=0)
        o1 = OrderedModel(fin.total.astype(int), fin[["mid", "z"]], distr="logit").fit(method="bfgs", disp=0)
        om = dict(b_mid_unadj=o0.params["mid"], p_mid_unadj=o0.pvalues["mid"], b_mid=o1.params["mid"], p_mid=o1.pvalues["mid"],
                  b_z=o1.params["z"], p_z=o1.pvalues["z"])
    except Exception as e:
        om = dict(error=str(e))
    print(pd.DataFrame(rows).to_string(index=False))
    print("KW（文字数×群）:", {k: round(v, 3) for k, v in kwn.items()})
    print("到達/未到達の文字数:\n", ach_n)
    print("相関:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cor.items()})
    print(ST.to_string(index=False))
    print(f"層化順列検定（文字数中央値{med}字で2層、{nperm}回）p = {p_strat:.4f}")
    print("ロジスティック（到達 ~ 中頻度群 + 文字数z）:", {k: round(v, 3) if isinstance(v, float) else v for k, v in lr.items()})
    print("順序ロジット（最終振り返り段階 ~ 中頻度群 + 文字数z）:", {k: round(v, 3) if isinstance(v, float) else v for k, v in om.items()})
    with open(OUT / "11_記述量の交絡.txt", "w", encoding="utf-8") as fh:
        fh.write(pd.DataFrame(rows).to_string(index=False) + "\n")
        fh.write(f"KW（最終振り返りの文字数×群）：H(2) = {kwn['H']:.3f}, p = {kwn['p']:.4f}\n")
        fh.write("到達(1)/未到達(0)の文字数:\n" + ach_n.round(1).to_string() + "\n")
        fh.write("相関: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in cor.items()}, ensure_ascii=False) + "\n")
        fh.write(ST.to_string(index=False) + "\n")
        fh.write(f"層化順列検定（文字数の中央値{med}字で2層に分け層内で群ラベルを並べ替え、統計量＝層別χ²の和、{nperm}回）：p = {p_strat:.4f}\n")
        fh.write("ロジスティック回帰（到達 ~ 広範・中頻度群ダミー [+ 文字数z]）: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in lr.items()}, ensure_ascii=False) + "\n")
        fh.write("順序ロジット（最終振り返り段階1〜4 ~ 広範・中頻度群ダミー [+ 文字数z]）: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in om.items()}, ensure_ascii=False) + "\n")
    S["LEN"] = dict(rows=rows, kw=kwn, ach_n=ach_n, cor=cor, ST=ST, p_strat=p_strat, med=med, lr=lr, om=om, nperm=nperm)


def sec_supp(c, child_total, S, B, rng):
    H1("[補足資料] 表14相当（L2ロジスティック回帰）・k=4解の内訳")
    a = child_total.merge(c[["child"] + FEATS], on="child")
    X = StandardScaler().fit_transform(a[FEATS])
    y = a.ach_last.astype(int).to_numpy()
    co = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000).fit(X, y).coef_[0]
    Bc = []
    for _ in range(B):
        idx = rng.integers(0, len(a), len(a))
        if len(set(y[idx])) < 2:
            continue
        Xb = StandardScaler().fit_transform(a[FEATS].to_numpy()[idx])
        Bc.append(LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000).fit(Xb, y[idx]).coef_[0])
    Bc = np.array(Bc)
    rows = []
    for j, (f, fj) in enumerate(zip(FEATS, FEATS_JP)):
        rho, p = stats.spearmanr(a[f], y)
        rows.append(dict(指標=fj, 係数=co[j], CI下限=np.percentile(Bc[:, j], 2.5), CI上限=np.percentile(Bc[:, j], 97.5),
                         正の割合=(Bc[:, j] > 0).mean(), 単変量rho=rho, 単変量p=p))
    LT = pd.DataFrame(rows)
    print("y = 最終振り返りでA以上（n = 28）、StandardScaler、L2、C = 1.0、lbfgs；ブートストラップ", len(Bc), "回")
    print(LT.round(3).to_string(index=False))
    LT.to_csv(OUT / "12_補足_表14相当_ロジスティック回帰.csv", index=False, encoding="utf-8-sig")
    # k=4
    lab4 = S["T7"]["labels4"]
    a4 = a.copy()
    a4["k4"] = a4.child.map(dict(zip(c.child, lab4)))
    k4 = a4.groupby("k4").agg(n=("child", "size"), 原群=("group", lambda s: "、".join(f"{k}{v}" for k, v in s.value_counts().items())),
                              最終振り返り到達率=("ach_last", "mean"), p9到達率=("ach_p9", "mean"), 変化量=("change", "mean"),
                              閲覧回数=("view_count", "mean"), 閲覧時間=("view_duration", "mean"), 平均閲覧時間=("view_mean_duration", "mean"),
                              参照の広さ=("view_breadth", "mean"), 参照の深さ=("view_depth", "mean"), 参照の集中度=("view_gini", "mean"))
    print(k4.round(3).to_string())
    k4.to_csv(OUT / "12_補足_k4解の内訳.csv", encoding="utf-8-sig")
    S["SUPP"] = dict(LT=LT, k4=k4, nboot=len(Bc))


def sec_holm(S):
    H1("[補足資料] 実施した検定の一覧とHolm補正後の p")
    L = []
    T11 = S["T11"]
    for col, lab in [("mean", "平均"), ("last", "最終振り返り"), ("max", "最大"), ("front", "前半"), ("back", "後半"), ("change", "変化量")]:
        L.append(dict(区分="本文", 分析=f"表11 KW：{lab}（原総合）", p=T11["kw"][col]["p"]))
    for g in ORDER + ["全体"]:
        L.append(dict(区分="本文", 分析=f"表11 Wilcoxon正確検定：{g}（原総合）", p=T11["wil"][g]["p_exact"]))
    L.append(dict(区分="本文", 分析="変化量の群×時期交互作用（線形混合モデルLRT）", p=T11["lmm"]["p"]))
    labs12 = {"ach_major": "① 平均ベース", "ach_last": "② 最終振り返りでA以上", "ach_any": "③ 1回以上A以上", "ach_front": "④ 前半",
              "ach_back": "⑤ 後半", "ach_p9": "② p9のみ（n = 19）"}
    for col, r in S["T12"]["res"].items():
        L.append(dict(区分="本文", 分析=f"表12 FFH：{labs12[col]}", p=r["p_ffh"]))
    L.append(dict(区分="本文", 分析="表10 ルーブリック×SOLO Spearman ρ", p=S["T10"]["out"]["p"]))
    for lab, k in S["T432"]["keep"].items():
        if lab.startswith("原総合") or lab.startswith("（"):
            continue
        L.append(dict(区分="本文4.3.2", 分析=f"{lab}：①平均KW", p=k["k1"]["p"]))
        L.append(dict(区分="本文4.3.2", 分析=f"{lab}：②最終振り返りFFH", p=k["r2"]["p_ffh"]))
        L.append(dict(区分="本文4.3.2", 分析=f"{lab}：②p9のみFFH", p=k["r9"]["p_ffh"]))
        for g in ORDER:
            L.append(dict(区分="本文4.3.2", 分析=f"{lab}：③{g} Wilcoxon", p=k["w"][g]["p_exact"]))
        L.append(dict(区分="本文4.3.2", 分析=f"{lab}：学級全体 Wilcoxon", p=k["wa"]["p_exact"]))
    D = pd.DataFrame(L)
    D["p"] = D["p"].astype(float).fillna(1.0)
    D["Holm補正後p"] = multipletests(D.p, method="holm")[1]
    D = D.sort_values("p").reset_index(drop=True)
    print(D.round(4).to_string())
    print("検定の件数:", len(D), " Holm補正後 p < .05:", int((D["Holm補正後p"] < .05).sum()))
    D.to_csv(OUT / "12_補足_全検定一覧_Holm.csv", index=False, encoding="utf-8-sig")
    S["HOLM"] = D


def sec_versions(args, S):
    v = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
             scipy=scipy.__version__, sklearn=sklearn.__version__, statsmodels=statsmodels.__version__, seed=SEED,
             B_resample=args.B, solo=args.solo, solo_file=args.solo_file, run_at=dt.datetime.now().isoformat(timespec="seconds"),
             fast=args.fast)
    with open(OUT / "00_versions.txt", "w", encoding="utf-8") as fh:
        for k, x in v.items():
            fh.write(f"{k}: {x}\n")
    print(v)
    S["VER"] = v


# =============================================================================
# 3b. 数値確定表.md の自動生成
# =============================================================================
def write_number_table(S):
    L = []
    w = L.append
    sl = S["solo_label"]
    provisional = "既存Claude" in sl
    T7, T8, T9, T10, T11, T12, T432, T433, LEN = (S[k] for k in ["T7", "T8", "T9", "T10", "T11", "T12", "T432", "T433", "LEN"])
    K = T7["K"].set_index("k")
    J = T7["J"].set_index("群")
    SE = T7["sens"].set_index("分析")
    kw = T11["kw"]
    wil = T11["wil"]
    r12 = T12["res"]
    keep = T432["keep"]
    gshort = {"広範・高頻度群": "広範・高頻度群", "広範・中頻度群": "広範・中頻度群", "限定・長時間群": "限定・長時間群"}

    def pc(r, i):
        a, n = r["counts"][i]
        return f"{_r(100 * a / n, 0)}%" if n else "—"

    def rate_txt(r, i):
        a, n = r["counts"][i]
        return f"{pc(r, i)}（{a}/{n}）" if n else "—"

    def rate_c(r, i):
        a, n = r["counts"][i]
        return f"{pc(r, i)}，{a}/{n}" if n else "—"

    def rates3(r, counts=True):
        if counts:
            return "，".join(f"{g} {rate_txt(r, i)}" for i, g in enumerate(ORDER))
        return "，".join(f"{g} {pc(r, i)}" for i, g in enumerate(ORDER))

    w("# 数値確定表（s2026-023 改稿用）")
    w("")
    w(f"- 生成：`査読対応/analysis/reproduce_all.py`（{S['VER']['run_at']}，乱数シード {SEED}，再標本化 B = {S['VER']['B_resample']}）")
    w(f"- 環境：Python {S['VER']['python']}，numpy {S['VER']['numpy']}，pandas {S['VER']['pandas']}，scipy {S['VER']['scipy']}，"
      f"scikit-learn {S['VER']['sklearn']}，statsmodels {S['VER']['statsmodels']}")
    w(f"- SOLOの出所：**{S['solo_desc']}**" + ("。**暫定値**（3LLM再評価の承認後に `--solo reeval` で再実行し、SOLOを含む箇所〔表10・4.3.2表のSOLO行・判定・4.2.3の文・記述量の相関〕を差し替える）" if provisional else ""))
    if S["VER"].get("fast"):
        w("- **注意：--fast（再標本化回数1/10）で生成した動作確認用。数値は確定値ではない。**")
    w("- 群名：広範・高頻度群（旧：網羅的探索群），広範・中頻度群（旧：均等群），限定・長時間群（旧：局所滞留群）")
    w("- 表記：原稿の句読点（，．）に合わせた。表・文中の統計記号は *p*, *H*, *M*, *SD*, *n*, *W*, *r*, *V*, *k* をイタリック，ε², ρ, κ, χ² は立体。")
    w("- 本ファイルは `reproduce_all.py` の実行ごとに上書きされる。手で編集した内容は残らないので，転記用の控えは別ファイルに取ること。")
    w("")

    # ---------------- 表2
    w("## 表2（改）他者参照に関する指標")
    w("")
    w(md_table(["軸", "指標", "説明", "算出式"], S["T2"]["defs"]))
    w("")
    info = S["T2"]["info"]
    chk = S["T2"]["check"]
    w(f"注：展開部（p4〜p9，{info['date_min']}〜{info['date_max']}のログ）で他の児童のワークシートを開いた操作{info['n_view_peer']:,}件"
      f"（閲覧操作{info['n_view_all']:,}件から教師のワークシートの閲覧{info['n_view_teacher']}件を除く）から算出した．"
      f"参照の広さが27名に達した児童は{info['breadth_27']}名であった．")
    w("")
    chk_txt = "，".join("{} {}/{}".format(x["指標"], x["一致数"], x["児童数"]) for x in chk[:6])
    w(f"- 再計算の確認：生ログ（`分析データ/logs_with_duration_v2.xlsx`）から6指標を再計算し，`clustering_results.csv` と"
      f"{chk_txt} で一致した（差 < 10⁻⁶）。"
      f"アーカイブのCSV版ログからの再計算も同一（{info['xlsx_csv_identical']}）。")
    w(f"- 参照の集中度を「分母27固定（未閲覧の児童を0回として27名で算出）」で計算すると一致は{chk[6]['一致数']}/28（参照の広さが27の児童のみ一致）。"
      f"→ 本文の定義を実際の算出（閲覧した児童の間の偏り）に改める（決定6）。")
    w("- 閲覧時間はログに付与済みの継続秒数（最大60秒，最小1秒）を合計した。原稿の「60秒超の操作空白で閲覧中断とみなす」はこの列の作成段階の処理で，本スクリプトでは再現していない（列の最大値が60秒であることのみ確認）。")
    w("")
    w("**3.3.1の文（案）**")
    w("")
    w(f"> 参照の集中度は，児童が閲覧した他者の間での閲覧回数の偏りを表すジニ係数であり，閲覧しなかった他者は算出に含めない．"
      f"なお，閲覧しなかった他者を0回として自分以外の27名で算出したジニ係数を用いても，*k* = 3の解は同一であった（ARI = {fz(SE.loc['参照の集中度を分母27固定で算出した6指標', 'ARI'], 2)}）．")
    w("")

    # ---------------- 表6
    w("## 表6　3群の特徴量の平均値（標準偏差）")
    w("")
    hdr = ["群", "*n*"] + FEATS_JP
    rows = [[r["群"], r["n"]] + [r[j] for j in FEATS_JP] for r in S["T6"]]
    w(md_table(hdr, rows))
    w("")
    w("注：6指標は類型化（k-means法）の入力変数であるため，群間の検定は行わない．標準偏差は不偏分散に基づく．")
    w("")

    # ---------------- 表7
    w("## 表7（新）クラスタ数の評価と類型の安定性")
    w("")
    w("**(a) クラスタ数別の評価指標（6指標・Z得点，k-means法，初期値200通りの最良解）**")
    w("")
    rows = []
    for k in range(2, 7):
        r = K.loc[k]
        rows.append([k, fz(r.silhouette), fn(r.CH, 2), fn(r.DB, 3), f"{fn(r.gap, 3)} ({fn(r.sk, 3)})", r.sizes])
    w(md_table(["*k*", "シルエット係数", "Calinski–Harabasz", "Davies–Bouldin", "Gap（*s*<sub>k</sub>）", "群の大きさ"], rows))
    w("")
    ksil = int(K.silhouette.idxmax())
    kch = int(K.CH.idxmax())
    kdb = int(K.DB.idxmin())
    w(f"注：シルエット係数とCalinski–Harabasz指標は大きいほど，Davies–Bouldin指標は小さいほど良い．Gap統計量は一様参照分布{T7['Bg']}回に基づき，"
      f"1-SE則（Gap(*k*) ≥ Gap(*k*+1) − *s*<sub>k+1</sub> を満たす最小の*k*）では *k* = {T7['k_1se']}．最良値：シルエット *k* = {ksil}，CH *k* = {kch}，DB *k* = {kdb}．")
    w("")
    w(f"**(b) *k* = 3の解の安定性（再標本化各{T7['stab']['B']:,}回）**")
    w("")
    rows = []
    for g in ORDER:
        r = J.loc[g]
        rows.append([g, int(r.n), fz(r.boot_mean, 2), f"{r.boot_lt05 * 100:.1f}%", fz(r.sub_mean, 2), f"{r.sub_lt05 * 100:.1f}%"])
    w(md_table(["群", "*n*", "ブートストラップ：平均Jaccard", "Jaccard < .5の割合", f"80%部分標本（{T7['stab']['n_sub']}名）：平均Jaccard", "Jaccard < .5の割合"], rows))
    w("")
    sens_main = ["導出2指標（平均閲覧時間/回・参照の深さ）を除いた4指標", [x for x in SE.index if x.startswith("主成分得点（累積寄与率80%")][0],
                 "Ward法（6指標）", "参照の集中度を分母27固定で算出した6指標"]
    stxt = "，".join(f"{n_.replace('（6指標）', '')} {fz(SE.loc[n_, 'ARI'], 2)}" for n_ in sens_main)
    w(f"注：Jaccard係数は群ごとに，元の群と再標本での最も重なる群との一致度（Hennig, 2007）．目安は .75以上で安定，.5未満で解消．"
      f"感度分析（*k* = 3の解と原解のARI）：{stxt}．")
    w("")
    w("**補足（本文に入れるかは任意）**")
    w("")
    st = T7["stab"]
    w(f"- 再標本の重心で全28名を割り当てたときの原解とのARI：ブートストラップ平均 {fz(st['boot_ari_mean'], 2)}（完全一致 {st['boot_ari_eq1'] * 100:.0f}%），"
      f"80%部分標本平均 {fz(st['sub_ari_mean'], 2)}（完全一致 {st['sub_ari_eq1'] * 100:.0f}%）。")
    bk = T7["bestk"]
    w("- 80%部分標本でシルエット係数が最大となった *k* の割合：" + "，".join(f"*k* = {int(k)} {v * 100:.1f}%" for k, v in bk.items()) + "。")
    low = T7["low_sil"]
    w("- 個体シルエットの最小値：" + "，".join(f"{r.child}（{r.group}）{fz(r.silhouette, 2)}" for r in low.itertuples()) + "。")
    other = SE[SE.index.str.startswith("（")]
    w("- その他の手法・特徴量（ARI，所属が変わった児童数）：" + "；".join(f"{re.sub(r'^（(参考|補足)）', '', i)} {fz(r.ARI, 2)}（{r['所属が変わった児童数']}名）" for i, r in other.iterrows()) + "。")
    w("")
    # 本文
    sp = T7["split"]

    def split_txt(k):
        parts = []
        for g in ORDER:
            ds = sp[k][g]
            if len(ds["parts"]) > 1:
                parts.append(f"{g}が{'名と'.join(map(str, ds['parts']))}名に分かれ")
        return "，".join(parts)
    small = {k: min(int(x) for x in K.loc[k, "sizes"].split("/")) for k in [5, 6]}
    intact4 = [g for g in ORDER if sp[4][g]["intact"]]
    pc80 = [x for x in SE.index if x.startswith("主成分得点（累積寄与率80%")][0]
    nc_txt = re.search(r"：(\d)成分，([\d.]+)%", pc80)
    jmin_g = J.boot_mean.idxmin()
    w("**3.3.1の文（案）**")
    w("")
    w(f"> クラスタ数は，*k* = 2〜6のそれぞれについてシルエット係数，Calinski–Harabasz指標，Davies–Bouldin指標およびGap統計量を算出して検討した．"
      f"k-means法の初期値依存を避けるため，初期値を変えた複数の試行から最良の解を採用した．各群の安定性は，復元抽出による再標本化（{st['B']:,}回）と"
      f"80%部分標本（{st['n_sub']}名，{st['B']:,}回）で得た群ごとのJaccard係数の平均により評価した [Hennig 2007]．"
      f"6指標には他の指標から算出される2指標（平均閲覧時間/回，参照の深さ）が含まれるため，この2指標を除いた4指標による再分析を感度分析として行った．")
    w("")
    w("**4.1の文（案）**")
    w("")
    sents = [f"シルエット係数は *k* = {ksil}で最大（{fz(K.loc[ksil, 'silhouette'])}）であった一方，Calinski–Harabasz指標は *k* = {kch}，"
             f"Davies–Bouldin指標は *k* = {kdb}，Gap統計量（1-SE則）は *k* = {T7['k_1se']}で最良となり，指標間で一致しなかった（表7(a)）．"]
    if len(intact4) == 2:
        sents.append(f"*k* = 4の解では{intact4[0]}と{intact4[1]}の構成は変わらず，{split_txt(4)}た．")
    else:
        sents.append(f"*k* = 4の解では{split_txt(4)}た．")
    sents.append(f"*k* = 5，6では{min(small.values())}〜{max(small.values()) if max(small.values()) > min(small.values()) else min(small.values()) + 0}名からなる群が生じた．".replace("1〜1名", "1名"))
    sents.append("そこで，シルエット係数が最大で，すべての群に解釈可能な特徴がある *k* = 3を採用した．")
    jtxt = "，".join(f"{g} {fz(J.loc[g, 'boot_mean'], 2)}" for g in ORDER)
    stxt2 = "，".join(fz(J.loc[g, "sub_mean"], 2) for g in ORDER)
    allok = (J.boot_mean >= .75).all()
    sents.append(f"再標本化による群別の平均Jaccard係数は{jtxt}であり（80%部分標本では順に{stxt2}），"
                 + ("いずれも安定とされる目安（.75）を上回った．" if allok else "一部の群で安定とされる目安（.75）を下回った．"))
    sents.append(f"ただし{jmin_g}は再標本の{J.loc[jmin_g, 'boot_lt05'] * 100:.0f}%で係数が .5を下回り，{int(J.loc[jmin_g, 'n'])}名からなる本群の境界は相対的に不安定であった．")
    ar4 = SE.loc[sens_main[0], "ARI"]
    arw = SE.loc["Ward法（6指標）", "ARI"]
    arp = SE.loc[pc80, "ARI"]
    chp = int(SE.loc[pc80, "所属が変わった児童数"])
    s_ = f"導出2指標を除いた4指標およびWard法でも同一の3群が得られ（いずれもARI = {fz(min(ar4, arw), 2)}）"
    if arp < 1:
        s_ += f"，累積寄与率{nc_txt.group(2)}%の主成分得点（{nc_txt.group(1)}成分）では{chp}名の所属が変わった（ARI = {fz(arp, 2)}）．"
    else:
        s_ += f"，累積寄与率{nc_txt.group(2)}%の主成分得点（{nc_txt.group(1)}成分）でも同一であった．"
    sents.append(s_)
    a, b = T7["pair"]
    sents.append(f"指標間の順位相関は{a}と{b}で最大（ρ = {fz(T7['rho_max'])}）であり，両指標のVIFはそれぞれ{T7['vif'][a]:,.0f}，{T7['vif'][b]:,.0f}であった．"
                 f"参照の広さが多くの児童で上限（27名）に近いため，参照の深さ（閲覧回数÷参照の広さ）が閲覧回数とほぼ同じ情報になっている．")
    sents.append("なお，6指標は類型化の入力変数であるため群間の検定は行わず，各群の特徴を記述統計で示す（表6）．")
    w("> " + "".join(sents))
    w("")

    # ---------------- 表8
    res = T8["res"]
    w("## 表8　LLM評価の信頼性指標（*n* = 50）")
    w("")
    w(md_table(["指標", "値"], T8["rows"]))
    w("")
    w(f"- 再現：κw，κ，F1，完全一致率，隣接一致率は `振り返り記述_50サンプル_層化抽出.csv`（第一著者の評定）と3LLM中央値から，ICC(2,3)とPearson相関は3モデルの総合判定（148記述）から再計算し，既存値と一致した。")
    notin = ", ".join("{} p{}".format(x["child"], x["page"]) for x in res["claude_not_in_rubric"])
    noclaude = ", ".join("{} p{}".format(x["child"], x["page"]) for x in res["rubric_without_claude"])
    w(f"- 人手再評定の母集団：ルーブリック評価の148記述のうち既存Claude SOLO評価がある**{res['pop_n']}記述**（Claude SOLOは{res['claude_rows']}記述で，"
      f"うち{len(res['claude_not_in_rubric'])}記述〔{notin}〕は148記述に含まれない。"
      f"148記述のうち{len(res['rubric_without_claude'])}記述〔{noclaude}〕はSOLO未評価）。"
      f"50記述はすべてこの母集団に含まれ，抽出率は **{res['rate'] * 100:.1f}%**（原稿の「148件・33.8%」は誤り）。層別の抽出数は抽出メタデータと一致した。")
    w("")
    w("**3.3.2(2)の文（案）**")
    w("")
    w(f"> 第三に，LLM評価の妥当性検証として，148記述のうち既存のSOLO評価がある{res['pop_n']}記述からSOLOレベルで層化無作為抽出した50記述（{res['rate'] * 100:.1f}%）について，"
      f"第一著者がLLMの判定を非開示の状態で再評定した．")
    w("")
    w("**4.2.1の文（案）**（値は既存どおり）")
    w("")
    w(f"> 3LLM中央値と人手再評定との一致度はCohen's κ<sub>w</sub> = {fz(res['kappa_w'])}（4段階・線形重み付け），二値化（A以上 vs B以下）でのCohen's κ = {fz(res['kappa_bin'])}であった．"
      f"3LLM相互の評価一致度はICC(2,3) = {fz(res['ICC2k'])}であった．")
    w("")

    # ---------------- 表9
    w("## 表9（改）振り返り記述の深まりの分布")
    w("")
    rows = [[r["区分"]] + [r[k] for k in ["S", "A", "B", "C"]] + [r["計"]] for r in T9["rows"][:2]]
    w(md_table(["区分", "S", "A", "B", "C", "計"], rows))
    w("")
    fpc = T9["fpcount"]
    w(f"注：人数（%）．児童ごとの平均段階は148記述の総合判定（S = 4〜C = 1）の平均を四捨五入した段階．最終振り返りは各児童が展開部で最後に記入した振り返りで，"
      f"第6場面（p9）{fpc.get(9, 0)}名，第5場面（p8）{fpc.get(8, 0)}名，第4場面（p7）{fpc.get(7, 0)}名である．")
    d9 = T9["d9"]
    w(f"（参考）p9（第6場面）に記述がある{T9['n9']}名に限った段階分布：S {d9['S'][0]}，A {d9['A'][0]}，B {d9['B'][0]}，C {d9['C'][0]}。表には入れず本文1文で示す。")
    w("")
    w("**4.2.2の文（案）**")
    w("")
    w(f"> 各児童の最終振り返り（第6場面{fpc.get(9, 0)}名，第5場面{fpc.get(8, 0)}名，第4場面{fpc.get(7, 0)}名）で単元目標を達成（A以上）した児童は{T9['n_last_A']}名（{T9['n_last_A'] / 28 * 100:.1f}%）であった．"
      f"第6場面（p9）に記述がある{T9['n_p9']}名に限ると{T9['n_p9_A']}名（{T9['n_p9_A'] / T9['n_p9'] * 100:.1f}%）であった．"
      f"単元期間中に1回以上A以上に到達した児童は{T9['n_any']}名（{T9['n_any'] / 28 * 100:.1f}%）であった．")
    w("")

    # ---------------- 表10
    o = T10["out"]
    ct = T10["ct"]
    w(f"## 表10　単元目標準拠ルーブリックとSOLOタクソノミーの対応［{sl}］")
    w("")
    rows = []
    for r in ["C", "B", "A", "S", "計"]:
        rows.append([r] + [int(ct.loc[r, k]) for k in [1, 2, 3, 4, 5]] + [int(ct.loc[r, "計"])])
    w(md_table(["ルーブリック ＼ SOLO", "1", "2", "3", "4", "5", "計"], rows))
    w("")
    w(f"注：*n* = {o['n']}記述．Spearmanの順位相関 ρ = {fz(o['rho'])}（*p* {fpe(o['p'])}），ρ² = {fz(o['rho2'], 2)}．"
      f"A以上 × 関係構造（Level 4）以上の二値一致 {o['n_agree']}/{o['n']}（{o['agree'] * 100:.1f}%），κ = {fz(o['kappa'], 2)}．")
    w("")
    Wt = T10["W"].set_index("段階")
    w("- 各ルーブリック段階内のSOLOの分布：" + "，".join(f"{r} Level {int(Wt.loc[r, '最小'])}〜{int(Wt.loc[r, '最大'])}（{int(Wt.loc[r, '段階数'])}段階，*SD* = {fn(Wt.loc[r, 'SD'])}）" for r in ["C", "B", "A", "S"]) + "。")
    w(f"- 不一致の2型：(1) SOLO関係構造以上・ルーブリックB以下 **{o['t1']}記述**（文字数の中央値{o['t1_nchar_med']:.0f}字），(2) ルーブリックA以上・SOLO多重構造以下 **{o['t2']}記述**（同{o['t2_nchar_med']:.0f}字）。全件は `out/06_T10補_不一致2型の記述一覧.csv`。")
    w(f"- 要素別のSOLOとの順位相関：要素1 {fz(o['el']['eA'], 2)}，要素2 {fz(o['el']['eB'], 2)}，要素3 {fz(o['el']['eC'], 2)}。")
    w(f"- 文字数との関係：ρ(SOLO, 文字数) = {fz(o['rho_solo_nchar'], 2)}，ρ(ルーブリック, 文字数) = {fz(o['rho_rub_nchar'], 2)}，文字数を統制した偏順位相関 ρ = {fz(o['prho'], 2)}。"
      f"文字数が下位25%（{o['q25']:.0f}字以下）でSOLO 4以上の記述は{o['short_solo4']}件（SOLO 4以上の最短は{o['solo4_min_nchar']}字）→ 原稿の「短い記述だが構造化されている」は削除。")
    w("")
    w("**4.2.3の文（案）**")
    w("")
    w(f"> 両指標の順位相関は ρ = {fz(o['rho'])}（*p* {fpe(o['p'])}，*n* = {o['n']}）であり，単元目標の達成度が高い記述ほど構造的複雑性も高い傾向があった．"
      f"一方，共有される順位の分散は約{o['rho2'] * 100:.0f}%（ρ² = {fz(o['rho2'], 2)}）にとどまり，各ルーブリック段階のなかでSOLOは{int(Wt['段階数'].min())}〜{int(Wt['段階数'].max())}段階に分布していた（表10）．"
      f"A以上と関係構造以上の二値判定は{o['n']}記述中{o['n_agree']}記述（{o['agree'] * 100:.1f}%）で一致し（κ = {fz(o['kappa'], 2)}），不一致は主に2つの型であった．"
      f"第一は，学習方法への気づきを因果的に述べるが，本文の根拠や場面と結び付いた心情の読み取りが不十分な記述であり，SOLOでは関係構造以上，ルーブリックではB以下と判定された（{o['t1']}記述）．"
      f"第二は，本文の描写を根拠に心情を簡潔に述べて単元目標を満たすが，構造としては単一の因果や列挙にとどまる記述であり，ルーブリックではA以上，SOLOでは多重構造以下と判定された（{o['t2']}記述）．"
      f"以上から，両指標は相互に関連しつつも，ルーブリックは単元目標に照らした内容の妥当性を，SOLOは記述の構造的複雑性を区別して捉えていると解釈した．")
    w("")
    w("※ 不一致2型の特徴づけ（第一・第二の型の文言）は既存Claude SOLOの事例（予備分析）に基づく定型文である。件数は自動で更新されるが，特徴づけが当てはまるかは "
      "`out/06_T10補_不一致2型の記述一覧.csv` を読んで確認すること（SOLOを差し替えたときは必ず）。")
    w("")

    # ---------------- 表11
    w("## 表11（新，旧表13を統合）3群別の振り返り記述の深まりスコアと前半→後半の変化")
    w("")
    T = T11["T"].set_index("群")
    rows = []
    for g in ORDER + ["全体"]:
        r = T.loc[g]
        rows.append([("学級全体" if g == "全体" else g), int(r.n), r["平均"], r["最終振り返り"], r["最大"], r["前半"], r["後半"], r["変化量Δ"],
                     fn(r.W, 1), fp(r["正確p"]), fz(r.r, 2)])
    w(md_table(["群", "*n*", "平均", "最終振り返り", "最大", "前半（p4〜p6）", "後半（p7〜p9）", "変化量Δ", "*W*", "正確*p*", "*r*"], rows))
    w("")
    ktxt = "；".join(f"{lab} *H*(2) = {fn(kw[col]['H'])}, *p* = {fp(kw[col]['p'])}, ε² = {fz(kw[col]['eps2'])}"
                     for col, lab in [("mean", "平均"), ("last", "最終振り返り"), ("max", "最大"), ("front", "前半"), ("back", "後半"), ("change", "変化量")])
    w(f"注：*M*（*SD*）．4 = S，3 = A，2 = B，1 = C．変化量Δ＝後半平均−前半平均．*W*，正確*p*，*r*は前半→後半のWilcoxon符号順位検定（ゼロ差を除外，同順位は平均順位，"
      f"差は小数第6位で丸めて同順位を判定，*p*は符号の全組合せに基づく正確値），*r* = |*Z*|/√*n*．"
      f"3群間のKruskal-Wallis検定：{ktxt}．ε² = *H*/(*n* − 1)．")
    w("")
    w("- KWの順列検定（20,000回）の *p*：" + "，".join(f"{lab} {fp(kw[col]['p_perm'])}" for col, lab in [("mean", "平均"), ("last", "最終"), ("max", "最大"), ("front", "前半"), ("back", "後半"), ("change", "変化量")]) + "。漸近の *p* と結論は同じ。")
    w("- 群内検定の非ゼロ差の人数と最小可能 *p*：" + "，".join(f"{g} {wil[g]['n_nonzero']}名（{fp(wil[g]['p_min'])}）" for g in ORDER) + "。")
    w("- 対応のある順位双列相関 *r*<sub>rb</sub>（参考）：" + "，".join(f"{('学級全体' if g == '全体' else g)} {fz(wil[g]['r_rb'], 2)}" for g in ORDER + ["全体"]) + "。")
    bt = T11["boot"]
    w("- 変化量の群間差（ブートストラップ95%区間）：" + "；".join(f"{k[0]}−{k[1]} {fs(v[0])} [{fs(v[1])}, {fs(v[2])}]" for k, v in bt.items()) + "。")
    lm = T11["lmm"]
    pw = T11["pw"]
    w(f"- 群×時期（前半/後半）の交互作用：線形混合モデル（記述単位148，児童のランダム切片，最尤法）の尤度比検定 χ²(2) = {fn(lm['chi2'])}，*p* = {fp(lm['p'])}。")
    w(f"- 検出力の目安：変化量の観測効果量 Cohen's *f* = {fn(pw['f'])} → 一元配置分散分析の近似で検出力 {fz(pw['power_anova'], 2)}，観測分布からの再標本化によるKWの検出力 {fz(pw['power_kw'], 2)}。"
      f"*n* = 6/18/4で検出力80%に必要な *f* ≈ {fn(pw['f80'])}，観測 *f* のままなら総数約{pw['N80']}名。")
    w("")
    w("**4.3.1の文（案）：平均的傾向**")
    w("")
    others = [kw[c_]["p"] for c_ in ["last", "max", "front", "back", "change"]]
    w(f"> 児童ごとの平均スコアに3群間の有意差は認められなかった（*H*(2) = {fn(kw['mean']['H'])}，*p* = {fp(kw['mean']['p'])}，ε² = {fz(kw['mean']['eps2'])}；表11）．"
      f"最終振り返り，最大値，前半，後半および変化量でも有意差は認められなかった（すべて *p* {'> .15' if min(others) > .15 else '≥ ' + fp(min(others))}）．")
    w("")
    hi, mid, lim = (wil[g] for g in ORDER)
    al = wil["全体"]
    w("**4.3.1の文（案）：群内の時間的変化（条件7）**")
    w("")
    w(f"> 前半（p4〜p6）から後半（p7〜p9）への変化をWilcoxon符号順位検定（正確検定）で検討した（表11）．"
      f"学級全体では後半の値が高く（Δ = {fs(al['delta_M'])}，*W* = {fn(al['W'], 1)}，*p* {fpe(al['p_exact'])}，*r* = {fz(al['r'], 2)}），"
      f"群別では広範・中頻度群で有意であった（Δ = {fs(mid['delta_M'])}，*W* = {fn(mid['W'], 1)}，*p* {fpe(mid['p_exact'])}，*r* = {fz(mid['r'], 2)}）．"
      f"一方，限定・長時間群の変化量（Δ = {fs(lim['delta_M'])}）は広範・中頻度群と同程度であり，変化量に3群間の有意差は認められなかった"
      f"（*H*(2) = {fn(kw['change']['H'])}，*p* = {fp(kw['change']['p'])}）．群×時期の交互作用も認められなかった（線形混合モデル，χ²(2) = {fn(lm['chi2'])}，*p* = {fp(lm['p'])}）．"
      f"広範・高頻度群と限定・長時間群では前後で差のあった児童がそれぞれ{hi['n_nonzero']}名，{lim['n_nonzero']}名であり，正確検定で得られる最小の *p* 値は{fp(hi['p_min'])}，{fp(lim['p_min'])}である．"
      f"したがって，群内検定の有意・非有意の違いは群の大きさを反映しており，前半から後半への深まりは学級全体に共通する傾向と解釈した．")
    w("")

    # ---------------- 表12
    w("## 表12（新）単元目標達成の定義別の群別達成率")
    w("")
    rows = []
    T12t = T12["T"]
    for _, r in T12t.iterrows():
        rows.append([r["定義"]] + [r[g] for g in ORDER] + [r["計"], fp(r["FFH_p"]), fz(r["CramerV"], 2)])
    w(md_table(["達成の定義"] + ORDER + ["計", "*p*（FFH）", "Cramér's *V*"], rows))
    w("")
    fpg = T12["fp_by_group"]
    fptxt = "，".join(f"{g}：p9 {int(fpg.loc[g].get(9, 0))}名・p8 {int(fpg.loc[g].get(8, 0))}名・p7 {int(fpg.loc[g].get(7, 0))}名" for g in ORDER)
    w(f"注：達成はA以上（S・Aを含む）．*p* はFisher–Freeman–Halton正確検定，Cramér's *V* はχ²（連続修正なし）から算出．"
      f"最終振り返りは各児童が展開部で最後に記入した振り返り（全体でp9 {fpc.get(9, 0)}名，p8 {fpc.get(8, 0)}名，p7 {fpc.get(7, 0)}名；{fptxt}）．"
      f"②p9のみは第6場面に記述がある{r12['ach_p9']['n']}名に限った感度分析．多重性の補正は行っていない．")
    w("")
    r2, r9 = r12["ach_last"], r12["ach_p9"]
    mids = [lab for (col, lab) in [("ach_major", "①"), ("ach_any", "③"), ("ach_front", "④"), ("ach_back", "⑤")] if r12[col]["rates"][1] > max(r12[col]["rates"][0], r12[col]["rates"][2])]
    w("**4.3.1の文（案）：単元目標の達成**")
    w("")
    lab5 = {"ach_major": "①平均ベース", "ach_any": "③1回以上A以上", "ach_front": "④前半", "ach_back": "⑤後半"}
    top = {c_: ORDER[int(np.argmax(r12[c_]["rates"]))] for c_ in lab5}
    mid_top = [lab5[c_][1:] for c_ in lab5 if r12[c_]["rates"][1] > max(r12[c_]["rates"][0], r12[c_]["rates"][2])]
    other_p = [r12[c_]["p_ffh"] for c_ in lab5]
    w(f"> 最終振り返りで単元目標を達成した児童の割合は，広範・中頻度群が{rate_txt(r2, 1)}であり，広範・高頻度群（{rate_c(r2, 0)}）と限定・長時間群（{rate_c(r2, 2)}）より高い傾向を示した"
      f"（Fisher–Freeman–Halton検定，*p* = {fp(r2['p_ffh'])}，Cramér's *V* = {fz(r2['V'], 2)}；表12）．"
      f"第6場面（p9）に記述がある{r9['n']}名に限ると，達成率は広範・中頻度群{rate_txt(r9, 1)}，広範・高頻度群{rate_txt(r9, 0)}，限定・長時間群{rate_txt(r9, 2)}であった（*p* = {fp(r9['p_ffh'])}，*V* = {fz(r9['V'], 2)}）．"
      f"一方，達成の定義を変えると群の順序は一致しなかった．広範・中頻度群の達成率が最も高かったのは最終振り返りのほか{'と'.join(mid_top) if mid_top else 'なし'}の定義であり，"
      f"1回以上A以上では広範・高頻度群が{rate_txt(r12['ach_any'], 0)}で最も高かった．最終振り返り以外の定義では，いずれも群間に有意な差は認められなかった（*p* = {fp(min(other_p))}〜{fp(max(other_p))}）．"
      f"最終振り返りへの注目は集計後に行った事後的なものであり，この傾向は探索的な所見として扱う．")
    w("")
    w(f"- 参考：各定義で達成率が最も高かった群：" + "，".join(f"{lab5[c_]} {top[c_]}" for c_ in lab5) + f"。広範・中頻度群が最高の定義は②に加えて{'・'.join(mids) if mids else 'なし'}。"
      f"原稿4.3.1の「他の達成定義でも均等群の優位が一貫」は削除する。")
    w("")

    # ---------------- 4.3.2
    w("## 4.3.2用（新設「他者参照の要素を含まない評価による確認」）の表")
    w("")
    rows = []
    order_labs = ["原総合（要素1〜3）", "要素1・2のみ（規則版）", "要素3のみ（参考）", T432["labS"]]
    for lab in order_labs:
        k = keep[lab]
        r2_, r9_ = k["r2"], k["r9"]
        rows.append([lab.replace(f"{sl}：", "SOLO：") if lab == T432["labS"] else lab,
                     fp(k["k1"]["p"]),
                     "／".join(f"{100 * a_ / n_:.0f}" for a_, n_ in r2_["counts"]) + f"（*n* = {r2_['n']}）",
                     fp(r2_["p_ffh"]), fz(r2_["V"], 2),
                     "／".join(f"{100 * a_ / n_:.0f}" for a_, n_ in r9_["counts"]) + f"（*n* = {r9_['n']}）",
                     fp(r9_["p_ffh"]),
                     "／".join(f"{fs(k['w'][g]['delta_M'])}（{fp(k['w'][g]['p_exact'])}）" for g in ORDER),
                     fp(k["k3"]["p"]),
                     f"{fs(k['wa']['delta_M'])}（{fp(k['wa']['p_exact'])}）"])
    w(md_table(["指標", "① 平均：KW *p*", "② 最終振り返りの到達率 %（高頻度／中頻度／長時間）", "② *p*（FFH）", "② *V*",
                "② p9のみの到達率 %", "② p9のみ *p*", "③ 群内Δ（正確*p*）高頻度／中頻度／長時間", "③ 変化量の群間差 KW *p*", "学級全体Δ（正確*p*）"], rows))
    w("")
    w("※ 原稿の表13は「③ 変化量の群間差 KW *p*」と「学級全体Δ」のみを③として載せる（群内検定は条件7の論法を避けるため表13には載せない）。")
    w("")
    labS2 = T432["labS2"]
    k2 = keep[labS2]
    w(f"注：到達の閾値は，原総合・要素1・2・要素3はA以上，SOLOは関係構造（Level 4）以上．要素1・2のみ（規則版）は，要素1・2がともにA→A，いずれかがC→C，それ以外→Bとして再構成した判定．"
      f"①はKruskal-Wallis検定，②はFisher–Freeman–Halton正確検定，③の群内はWilcoxon符号順位検定（正確検定），③の群間差は変化量のKruskal-Wallis検定．"
      f"SOLOは{S['solo_desc']}" + (f"で，148記述中{keep[T432['labS']]['a'].n_rec.sum()}記述に評価がある．SOLOの最終振り返りは他の指標と同じ記述（同じページ）で判定し，その記述にSOLO評価がない児童（児童10，p9）は②から除いた．"
                                     f"直前の記述で代替すると到達率は{'／'.join(f'{100 * a_ / n_:.0f}' for a_, n_ in k2['r2']['counts'])}%（*n* = {k2['r2']['n']}，*p* = {fp(k2['r2']['p_ffh'])}，*V* = {fz(k2['r2']['V'], 2)}）．" if provisional else "．"))
    w("")
    cA, cB = T432["cA"], T432["cB"]
    w(f"**事前に固定した判定基準による判定：{T432['scenario']}**" + ("（暫定：SOLOが中立基準の3LLM再評価になった時点で確定）" if provisional else ""))
    w("")
    w(f"- 基準：S1＝要素1・2判定と中立SOLOの両方で，最終振り返りの到達率が広範・中頻度群で最高かつCramér's *V* ≥ .30。S2＝一方のみ成立，または方向は一致するが *V* < .30。S3＝両指標で広範・中頻度群が最高でない。*p* は判定に用いない（修正方針§5.2）。")
    w(f"- 要素1・2：広範・中頻度群が最高＝{'○' if cA['mid_highest'] else '×'}，*V* = {fz(cA['V'], 2)}（{'≥' if cA['V_ok'] else '<'} .30）→ {'成立' if cA['met'] else '不成立'}")
    w(f"- {sl}：広範・中頻度群が最高＝{'○' if cB['mid_highest'] else '×'}，*V* = {fz(cB['V'], 2)}（{'≥' if cB['V_ok'] else '<'} .30）→ {'成立' if cB['met'] else '不成立'}"
      f"（SOLO欠損を直前の記述で代替した場合の判定も{T432['scenario_alt']}）")
    eAB, eC_, eS = keep["要素1・2のみ（規則版）"], keep["要素3のみ（参考）"], keep[T432["labS"]]
    w("")
    w("**3.3.3の文（案）：確認の設計**（RQは1文とし，ここで確認事項として説明する：決定記録 2026-10-17）")
    w("")
    w("> 要素3（他者参照を踏まえた自己の解釈の統合）は他者参照に関わる内容を評価するため，他者参照行動の類型との対応を検討すると，評価の定義そのものが対応を生む可能性がある．"
      "そこで，他者参照の要素を含まない評価として，要素1・2のみから再構成した判定（両要素がA→A，いずれかがC→C，それ以外→B）とSOLOタクソノミー（関係構造以上を到達とする）を用いて，"
      "平均の比較（①），最終振り返りでの到達（②）および前半から後半への変化（③）を再分析した．"
      "対応が維持されたとみなす基準は分析前に定め，両指標で最終振り返りの到達率が広範・中頻度群で最も高く，かつCramér's *V* が .30以上である場合とした．")
    w("")
    w("**4.3.2の文（案）**")
    w("")
    dchg = eAB["w"]["広範・中頻度群"]
    same = eAB["r2"]["counts"] == keep["原総合（要素1〜3）"]["r2"]["counts"]
    w(f"> 要素1・2のみで再構成した判定でも，最終振り返りでの到達率は{rates3(eAB['r2'])}であり"
      + ("，原指標と同一であった" if same else "") +
      f"（*p* = {fp(eAB['r2']['p_ffh'])}，*V* = {fz(eAB['r2']['V'], 2)}；表X）．"
      f"前半から後半への変化も，広範・中頻度群（Δ = {fs(dchg['delta_M'])}，*p* {fpe(dchg['p_exact'])}）と学級全体（Δ = {fs(eAB['wa']['delta_M'])}，*p* {fpe(eAB['wa']['p_exact'])}）で原指標とほぼ同じであった．"
      + (f"SOLOタクソノミーでも，最終振り返りで関係構造以上に到達した児童の割合は広範・中頻度群で最も高かった（{rates3(eS['r2'], False)}；*p* = {fp(eS['r2']['p_ffh'])}，*V* = {fz(eS['r2']['V'], 2)}）．"
         if cB["mid_highest"] else
         f"SOLOタクソノミーでは，最終振り返りで関係構造以上に到達した児童の割合が最も高かったのは{ORDER[int(np.nanargmax(eS['r2']['rates']))]}であった（{rates3(eS['r2'], False)}；*p* = {fp(eS['r2']['p_ffh'])}，*V* = {fz(eS['r2']['V'], 2)}）．") +
      f"一方，要素3のみで判定すると，群との対応は{'最も弱かった' if (eC_['r2']['V'] <= min(eAB['r2']['V'], eS['r2']['V'] if not isnan(eS['r2']['V']) else 9)) else '弱かった'}（{rates3(eC_['r2'], False)}；*p* = {fp(eC_['r2']['p_ffh'])}，*V* = {fz(eC_['r2']['V'], 2)}）．"
      + (f"平均の比較では，いずれの指標でも3群間の有意差は認められなかった（*p* = {fp(min(eAB['k1']['p'], eS['k1']['p']))}〜{fp(max(eAB['k1']['p'], eS['k1']['p']))}）．"
         if min(eAB['k1']['p'], eS['k1']['p']) >= .05 else
         f"平均の比較では，要素1・2で *p* = {fp(eAB['k1']['p'])}，SOLOで *p* = {fp(eS['k1']['p'])} であった．")
      + ("分析前に定めた基準に照らすと，最終振り返りでの到達の傾向は他者参照の要素を含まない評価でも認められた．" if T432["scenario"].startswith("S1") else
         "分析前に定めた基準に照らすと，最終振り返りでの到達の傾向は他者参照の要素を含まない評価では部分的にしか認められなかった．" if T432["scenario"].startswith("S2") else
         "分析前に定めた基準に照らすと，最終振り返りでの到達の傾向は他者参照の要素を含まない評価では認められなかった．"))
    w("")
    w(f"- 補足（回答書用）：記述単位では「A以上」の判定と「要素1・2がともにA」が148記述中146記述で一致する（要素3がA以上／B以下の境界を左右したのは2記述）。このため②は原指標と同一になる。")
    w("")

    # ---------------- 4.3.3
    w("## 4.3.3用（最終振り返りの記述の特徴）")
    w("")
    T3 = T433["T"]
    rows = [[r["到達"], r["群"], r["n"], r["要素1"], r["要素2"], r["要素3"]] for _, r in T3.iterrows()]
    w(md_table(["最終振り返り", "群", "*n*", "要素1（本文根拠）A/B/C", "要素2（心情・情景）A/B/C", "要素3（他者参照を踏まえた統合）A/B/C"], rows))
    w("")
    w("注：各要素の3LLM中央値の判定の人数（A/B/C）．引用候補の本文・要素別判定・代表的な判定理由（3LLM×3回のうち要素判定が中央値と最も一致する試行の理由）は `out/10_T433_最終振り返り一覧_引用候補.csv`（児童の姓は〇〇に置換済み）．")
    w("")
    w("**4.3.3の文（案）**（引用例は第一著者が上記CSVから選ぶ）")
    w("")
    nonach = T3[(T3["到達"].str.startswith("B")) & (T3["群"] == "計")].iloc[0]
    ab_txt = "全員がA" if (T433["eA_all_A"] == T433["n_ach"] and T433["eB_all_A"] == T433["n_ach"]) else "Aがそれぞれ{}名，{}名".format(T433["eA_all_A"], T433["eB_all_A"])
    w(f"> 最終振り返りでA以上に到達した{T433['n_ach']}名では，要素1（本文根拠）と要素2（心情・情景の読み取り）は{ab_txt}であり，"
      f"要素3（他者参照を踏まえた解釈の統合）がAであったのは{T433['eC_A']}名であった（広範・中頻度群の到達者{T433['mid_ach']}名では{T433['mid_ach_eC_A']}名）．"
      f"未到達の{int(nonach['n'])}名では，要素2がAの児童はおらず（A/B/C = {nonach['要素2']}），要素1はA/B/C = {nonach['要素1']}であった．")
    w("")
    w(f"- 原稿4.3.2「達成12名に①②③の3要素が共通」は誤り：③（要素3）がAは広範・中頻度群の到達者{T433['mid_ach']}名中{T433['mid_ach_eC_A']}名。")
    w("")

    # ---------------- 記述量
    w("## 記述量（文字数）の交絡の確認")
    w("")
    rows = [[r["群"], r["n"], r["文字数_MSD"], f"{r['中央値']:.1f}"] for r in LEN["rows"]]
    w(md_table(["群", "*n*", "最終振り返りの文字数 *M*（*SD*）", "中央値"], rows))
    w("")
    an = LEN["ach_n"]
    cor = LEN["cor"]
    lr = LEN["lr"]
    om = LEN["om"]
    ST = LEN["ST"]
    w(f"- 群間（KW）：*H*(2) = {fn(LEN['kw']['H'])}，*p* = {fp(LEN['kw']['p'])}。到達者 {an.loc[1, 'mean']:.1f}字（*SD* = {an.loc[1, 'std']:.1f}） vs 未到達者 {an.loc[0, 'mean']:.1f}字（*SD* = {an.loc[0, 'std']:.1f}）。")
    w(f"- 相関：記述単位で ρ(文字数, ルーブリック) = {fz(cor['rho_nchar_total_rec'], 2)}（148記述），ρ(文字数, SOLO) = {fz(cor['rho_nchar_solo_rec'], 2)}（{cor['n_solo']}記述）；"
      f"最終振り返りで ρ(文字数, ルーブリック) = {fz(cor['rho_nchar_total_final'], 2)}，ρ(文字数, SOLO) = {fz(cor['rho_nchar_solo_final'], 2)}。")
    w(f"- 文字数の中央値（{LEN['med']}字）で層化した到達率：" + "；".join(f"{r['層']}（*n* = {r['n']}）" + "，".join(f"{g} {r[g]}" for g in ORDER) for _, r in ST.iterrows())
      + f"。層内で群ラベルを並べ替えた層化順列検定 *p* = {fp(LEN['p_strat'])}（{LEN['nperm']:,}回）。")
    if "b_mid" in lr:
        w(f"- ロジスティック回帰（到達 ~ 広範・中頻度群ダミー）：*b* = {fn(lr['b_mid_unadj'])}，*p* = {fp(lr['p_mid_unadj'])} → 文字数（*z*）を加えると *b* = {fn(lr['b_mid'])}，*p* = {fp(lr['p_mid'])}（文字数 *b* = {fn(lr['b_z'])}，*p* = {fp(lr['p_z'])}）。")
    if "b_mid" in om:
        w(f"- 順序ロジット（最終振り返り段階1〜4 ~ 広範・中頻度群ダミー）：*b* = {fn(om['b_mid_unadj'])}，*p* = {fp(om['p_mid_unadj'])} → 文字数を加えると *b* = {fn(om['b_mid'])}，*p* = {fp(om['p_mid'])}。")
    w("")
    w("**5章（限界）に書ける文（案）**")
    w("")
    w(f"> 第一に，最終振り返りの記述量には群による違いがあり（平均で広範・高頻度群{LEN['rows'][0]['文字数_MSD'].split()[0]}字，広範・中頻度群{LEN['rows'][1]['文字数_MSD'].split()[0]}字，限定・長時間群{LEN['rows'][2]['文字数_MSD'].split()[0]}字），"
      f"到達者の記述は未到達者より長かった（{_r(an.loc[1, 'mean'], 0)}字，{_r(an.loc[0, 'mean'], 0)}字）．"
      f"文字数を共変量に加えると，広範・中頻度群と到達との関連は同じ方向のまま有意ではなくなった．"
      f"記述量の多さは深まりの表れとも評価の交絡とも解釈でき，本研究のデータでは両者を区別できない．")
    w("")
    w(f"> （短縮版）最終振り返りでの到達の群差は記述量の差と重なっており（文字数を統制すると関連は有意でなくなった），記述量が深まりの表れか交絡かは本研究のデータでは区別できない．")
    w("")

    # ---------------- 補足資料
    w("## 補足資料（本文には入れない）")
    w("")
    LT = S["SUPP"]["LT"]
    rows = [[r["指標"], fs(r["係数"], 3), f"[{fs(r['CI下限'])}, {fs(r['CI上限'])}]", fz(r["単変量rho"], 3), fp(r["単変量p"])] for _, r in LT.iterrows()]
    w(f"**表14相当**：最終振り返りでA以上（*n* = 28，到達14名）を目的変数，6指標をStandardScalerで標準化，L2正則化（*C* = 1.0，lbfgs）。ブートストラップ{S['SUPP']['nboot']}回のパーセンタイル95%区間。")
    w("")
    w(md_table(["指標", "係数", "95%区間", "単変量ρ（到達）", "*p*"], rows))
    w("")
    w(f"→ 本文から削除し，必要なら1文：「6指標のいずれも単独では到達と明確な関連を示さなかった（|ρ| ≤ {fz(LT['単変量rho'].abs().max(), 2)}）」。")
    w("")
    k4 = S["SUPP"]["k4"]
    rows = [[i, int(r.n), r["原群"], f"{r['最終振り返り到達率'] * 100:.0f}%", f"{r['p9到達率'] * 100:.0f}%", fs(r["変化量"]), fn(r["閲覧回数"], 1), fn(r["平均閲覧時間"], 1), fn(r["参照の広さ"], 1), fn(r["参照の集中度"], 3)] for i, r in k4.iterrows()]
    w("**k = 4の解の内訳**")
    w("")
    w(md_table(["クラスタ", "*n*", "原群", "最終振り返り到達率", "p9到達率", "変化量", "閲覧回数", "平均閲覧時間/回", "参照の広さ", "参照の集中度"], rows))
    w("")
    w("→ 広範・中頻度群が2分割され，到達率が高い下位群は参照の集中度が高い（偏りのある）側である。「均等な参照」の解釈を主張から外す根拠（群名変更済み）。")
    w("")
    D = S["HOLM"]
    w(f"**検定の件数と多重性**：本文と4.3.2表で報告する検定は計{len(D)}件（`out/12_補足_全検定一覧_Holm.csv`）。Holm補正後に *p* < .05 となるのは{int((D['Holm補正後p'] < .05).sum())}件："
      + "；".join(f"{r['分析']}（*p* {fpe(r['p'])} → {fp(r['Holm補正後p'])}）" for _, r in D[D['Holm補正後p'] < .05].iterrows()) + "。"
      f"表12の②（*p* = {fp(r2['p_ffh'])}）は補正後 {fp(D.loc[D['分析'] == '表12 FFH：② 最終振り返りでA以上', 'Holm補正後p'].iloc[0])}。")
    w("")

    # ---------------- 旧→新
    w("## 原稿v4からの数値・記述の変更一覧（旧→新）")
    w("")
    rows = []
    rows.append(["表2・3.3.1 ジニ係数の定義", V4["gini_def"], "閲覧した児童の間の偏り（実際の算出）．分母27版は感度分析で同一解（ARI = 1.00）"])
    rows.append(["表2", "集中度の行なし（5行）", "参照の集中度の行を追加（6行）"])
    rows.append(["群名", "網羅的探索群／均等群／局所滞留群", "広範・高頻度群／広範・中頻度群／限定・長時間群"])
    rows.append(["表7", V4["T7_KW"], "削除し「クラスタ数の評価と安定性」の表に置換（表7(a)(b)）"])
    rows.append(["3.3.1 効果量", "ε²（実際は η²<sub>H</sub> の式）", "ε² = *H*/(*n*−1) に統一"])
    rows.append(["3.3.2(2)・付録 人手再評定", V4["T8_pop"], f"既存SOLO評価がある{res['pop_n']}記述から50記述（{res['rate'] * 100:.1f}%）"])
    rows.append(["表9 2行目", V4["T9_row2"], "「最終振り返りの段階」に改名（値は同じ）．p9のみは S 1，A 7，B 9，C 2（*n* = 19）"])
    rows.append(["4.2.2", V4["s422"], f"最終振り返りで{T9['n_last_A']}名（50.0%）．p9のみは{T9['n_p9']}名中{T9['n_p9_A']}名（{T9['n_p9_A'] / T9['n_p9'] * 100:.1f}%）"])
    rows.append(["4.2.3 解釈", "独立した側面からの多面的評価／「短い記述だが構造化されている」", f"相互に関連しつつ区別される側面（ρ² = {fz(o['rho2'], 2)}，二値一致{o['agree'] * 100:.1f}%，κ = {fz(o['kappa'], 2)}，不一致{o['t1']}件・{o['t2']}件）．「短いが構造化」は該当{o['short_solo4']}件のため削除"])
    rows.append(["表11・4.3.1 KW 平均", V4["T11_KW_mean"], f"*H*(2) = {fn(kw['mean']['H'])}, *p* = {fp(kw['mean']['p'])}, ε² = {fz(kw['mean']['eps2'])}"])
    rows.append(["4.3.1 KW その他", V4["T11_KW_others"], "最終 " + fp(kw['last']['p']) + "，最大 " + fp(kw['max']['p']) + "，前半 " + fp(kw['front']['p']) + "，後半 " + fp(kw['back']['p']) + "，変化量 " + fp(kw['change']['p'])])
    rows.append(["表12・要旨・5章", V4["T12"], f"FFH正確 *p* = {fp(r2['p_ffh'])}, *V* = {fz(r2['V'], 2)}（最終振り返り）；p9のみ *p* = {fp(r9['p_ffh'])}, *V* = {fz(r9['V'], 2)}（*n* = {r9['n']}）"])
    rows.append(["表12 表題", "最終ページp.9", "最終振り返り（p9 19名・p8 8名・p7 1名）"])
    rows.append(["4.3.1", V4["s431_consistent"], f"誤り．③1回以上A以上は広範・高頻度群 {rate_txt(r12['ach_any'], 0)} が最高．5定義を全数提示（表12）"])
    for g, key in zip(ORDER, ["T13_hi", "T13_mid", "T13_lim"]):
        rows.append([f"表13→表11 {g}", V4[key], f"*W* = {fn(wil[g]['W'], 1)}, 正確 *p* = {fp(wil[g]['p_exact'])}, *r* = {fz(wil[g]['r'], 2)}"])
    rows.append(["4.3.1・5章（新規）", "—", f"学級全体 Δ = {fs(al['delta_M'])}, *W* = {fn(al['W'], 1)}, *p* {fpe(al['p_exact'])}；変化量の群間差 *p* = {fp(kw['change']['p'])}；交互作用 *p* = {fp(lm['p'])}"])
    rows.append(["要旨", V4["abs_within"], f"Reflections deepened across the class (Δ = {fs(al['delta_M'])}, exact *p* {fpe(al['p_exact'])}), with no between-group difference in change (*p* = {fp(kw['change']['p'])})"])
    rows.append(["要旨", V4["abs_climax"], f"In the final reflection: {rate_txt(r2, 1).split('（')[0]} vs {rate_txt(r2, 0).split('（')[0]}/{rate_txt(r2, 2).split('（')[0]} (Fisher–Freeman–Halton *p* = {fp(r2['p_ffh'])}, *V* = {fz(r2['V'], 2)})"])
    rows.append(["表14・4.3.1・5章", V4["T14"] + "／" + V4["s5_pred"], f"本文から削除（単変量 ρ = {fz(LT.set_index('指標').loc['参照の広さ', '単変量rho'], 3)}，ブートストラップ区間が0をまたぐ）．補足資料へ"])
    rows.append(["4.3.2", V4["s432"], f"誤り．要素3がAは到達14名中{T433['eC_A']}名（広範・中頻度群12名中{T433['mid_ach_eC_A']}名）"])
    rows.append(["4.1（新規）", "シルエット係数のみ（0.429）", f"*k* = 2〜6の4指標，Jaccard {', '.join(fz(J.loc[g, 'boot_mean'], 2) for g in ORDER)}，4指標・Ward・分母27でARI = 1.00"])
    rows.append(["4.3.2（新設）", "—", f"要素1・2のみで到達率同一（{'／'.join(f'{100 * a_ / n_:.0f}' for a_, n_ in eAB['r2']['counts'])}%），SOLO {'／'.join(f'{100 * a_ / n_:.0f}' for a_, n_ in eS['r2']['counts'])}%，要素3のみ {'／'.join(f'{100 * a_ / n_:.0f}' for a_, n_ in eC_['r2']['counts'])}%．判定 {T432['scenario']}"])
    w(md_table(["箇所", "旧（v4）", "新"], rows))
    w("")
    w("変わらない数値：表6（6指標の群別平均・SD），シルエット係数 .429（*k* = 3），表8（κw .606，κ .595，F1 .773，完全一致70.0%，隣接一致98.0%，ICC(2,3) .895，*r* .72〜.87），"
      "表9の値，表10のクロス表と ρ = .615（*n* = 143），表11の記述統計，表12の達成者数（1/6，12/18，1/4）と *V* = .450。")
    with open(NUMBER_TABLE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# =============================================================================
# 4. main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", choices=["claude", "reeval"], default="claude", help="SOLOの出所（既定：既存Claude評価）")
    ap.add_argument("--solo-file", default=None, help="--solo reeval のとき：solo_reeval.py の集約結果CSV（3LLM中央値_SOLO列）")
    ap.add_argument("--fast", action="store_true", help="再標本化回数を1/10にした動作確認")
    ap.add_argument("--no-table", action="store_true", help="数値確定表.md を生成しない")
    args = ap.parse_args()
    if args.solo == "reeval" and not args.solo_file:
        ap.error("--solo reeval には --solo-file が必要です")
    args.B = 100 if args.fast else 1000
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logf)
    S = {}
    sec_versions(args, S)

    c = load_cluster()
    d = load_rubric()
    t = load_text()
    raw = load_raw()
    solo, solo_label, solo_desc = load_solo(args.solo, args.solo_file)
    S["solo_label"], S["solo_desc"] = solo_label, solo_desc
    pg = d.merge(t, on=["child", "page"], how="left").merge(solo, on=["child", "page"], how="left")
    assert pg.text.notna().all() and len(pg) == 148
    final_page = pg.groupby("child").page.max().to_dict()
    S["final_page"] = final_page
    print("記述数:", len(pg), " SOLOあり:", int(pg.solo.notna().sum()), " 児童数:", pg.child.nunique())
    print("ページ別記述数:", pg.page.value_counts().sort_index().to_dict())
    print("SOLO欠損の記述:", pg.loc[pg.solo.isna(), ["child", "page"]].to_dict("records"))

    sec_T2(c, S)
    sec_T6(c, S)
    sec_T7(c, S, args.B, args.fast)
    sec_T8(d, S)
    child_total = aggregate(pg, "total", 3, final_page).merge(c[["child", "group"]], on="child")
    assert len(child_total) == 28 and child_total.front.notna().all() and child_total.back.notna().all()
    child_solo = aggregate(pg, "solo", 4, final_page).merge(c[["child", "group"]], on="child")
    sec_T9(child_total, S)
    sec_T10(pg, S, solo_label)
    rng = np.random.default_rng(SEED + 11)
    sec_T11(child_total, pg, c, S, args.B, rng)
    sec_T12(child_total, S)
    sec_T432(pg, c, final_page, S, solo_label)
    sec_T433(pg, child_total, raw, c, S)
    sec_length(pg, child_total, child_solo, c, S, args.B, np.random.default_rng(SEED + 13))
    sec_supp(c, child_total, S, args.B, np.random.default_rng(SEED + 14))
    sec_holm(S)

    # 児童単位の統合データ
    master = c[["child", "group"] + FEATS].copy()
    master["旧群名"] = master.group.map(OLD_NAME)
    master["最終振り返りページ"] = master.child.map(final_page)
    for nm, df in [("総合", child_total), ("SOLO", child_solo)]:
        sub = df.drop(columns=["group"]).set_index("child").add_prefix(f"{nm}_")
        master = master.merge(sub, left_on="child", right_index=True, how="left")
    master.to_csv(OUT / "13_児童単位_統合データ.csv", index=False, encoding="utf-8-sig")
    pgo = pg.copy()
    pgo["text"] = pgo.text.map(mask_names)
    pgo.to_csv(OUT / "13_記述単位_統合データ.csv", index=False, encoding="utf-8-sig")

    if not args.no_table:
        write_number_table(S)
        print(f"\n数値確定表を出力: {NUMBER_TABLE}")
    print("\n完了。出力フォルダ:", OUT)


if __name__ == "__main__":
    main()
