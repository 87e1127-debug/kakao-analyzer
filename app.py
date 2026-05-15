"""카카오톡 대화 분석기 (Streamlit) — CSV 또는 카카오톡 TXT보내기."""

from __future__ import annotations

import io
import html
import os
import platform
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

_APP_DIR = Path(__file__).resolve().parent
_FONT_DIR = _APP_DIR / "fonts"
_PRIMARY_BUNDLED_FONT = _FONT_DIR / "NanumGothic-Regular.ttf"


_HANGUL_SYL_START = 0xAC00
_HANGUL_SYL_END = 0xD7A3


def _pil_truetype_ok(path: str) -> bool:
    """PIL로 TTF/OTF/TTC 열기 검사."""
    try:
        from PIL import ImageFont

        ImageFont.truetype(path, 40)
        return True
    except OSError:
        return False


def _font_cmap_contains_hangul(path: str) -> bool:
    """cmap에 한글 음절이 있는지 확인(잘못된/영문 전용 파일이 Nanum 이름으로 올라온 경우 제외)."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return True

    ext = Path(path).suffix.lower()
    try:
        if ext == ".ttc":
            for i in range(24):
                try:
                    font = TTFont(path, fontNumber=i)
                except Exception:
                    break
                cmap = font.getBestCmap()
                if cmap and any(_HANGUL_SYL_START <= cp <= _HANGUL_SYL_END for cp in cmap):
                    return True
            return False
        font = TTFont(path, fontNumber=0)
        cmap = font.getBestCmap()
        if not cmap:
            return False
        return any(_HANGUL_SYL_START <= cp <= _HANGUL_SYL_END for cp in cmap)
    except Exception:
        return True


def _font_ok_for_korean_wordcloud(path: str) -> bool:
    return _pil_truetype_ok(path) and _font_cmap_contains_hangul(path)


def materialize_font_for_cloud(src: str) -> tuple[str, list[str]]:
    """읽기 전용 볼륨 등에서 안정적으로 열리도록 임시 디렉터리에 복사. (경로, 삭제할 임시파일 목록)."""
    cleanup: list[str] = []
    try:
        raw = Path(src).read_bytes()
    except OSError:
        return src, cleanup
    if len(raw) < 4096:
        return src, cleanup
    suf = Path(src).suffix.lower() or ".ttf"
    fd, tmp = tempfile.mkstemp(prefix="wcfont_", suffix=suf)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return src, cleanup
    cleanup.append(tmp)
    return tmp, cleanup


def bundled_wordcloud_font_path() -> str | None:
    """프로젝트 `fonts/` 내 TTF. 나눔고딕 파일명 우선, 없으면 같은 폴더의 첫 번째 .ttf."""
    candidates: list[Path] = []
    if _PRIMARY_BUNDLED_FONT.is_file():
        candidates.append(_PRIMARY_BUNDLED_FONT)
    if _FONT_DIR.is_dir():
        for p in sorted(_FONT_DIR.glob("*.ttf")):
            if p.is_file() and p.resolve() not in {c.resolve() for c in candidates}:
                candidates.append(p)
        for p in sorted(_FONT_DIR.glob("*.otf")):
            if p.is_file():
                candidates.append(p)
    for p in candidates:
        s = str(p.resolve())
        if _font_ok_for_korean_wordcloud(s):
            return s
    return None


def system_korean_font_path() -> str | None:
    """OS 기본 한글 폰트(번들이 없을 때). Linux는 Streamlit / Debian 계열 경로 포함."""
    sysname = platform.system()
    if sysname == "Windows":
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "malgun.ttf")
        return p if os.path.isfile(p) and _font_ok_for_korean_wordcloud(p) else None
    if sysname == "Darwin":
        for p in (
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/AppleGothic.ttf",
        ):
            if os.path.isfile(p) and _font_ok_for_korean_wordcloud(p):
                return os.path.abspath(p)
        return None
    for p in (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.isfile(p) and _font_ok_for_korean_wordcloud(p):
            return os.path.abspath(p)
    return None


def iter_wordcloud_font_candidates(explicit: str | None) -> list[str]:
    """워드클라우드에 쓸 font_path 후보(앞이 우선). 번들 → 명시 경로 → 시스템."""
    seen: set[str] = set()
    out: list[str] = []

    def add(p: str | None) -> None:
        if not p:
            return
        ap = os.path.abspath(p) if os.path.isfile(p) else p
        if not os.path.isfile(ap):
            return
        if ap in seen:
            return
        if not _font_ok_for_korean_wordcloud(ap):
            return
        seen.add(ap)
        out.append(ap)

    add(bundled_wordcloud_font_path())
    if explicit:
        ep = os.path.abspath(os.path.expanduser(str(explicit).strip()))
        if os.path.isfile(ep):
            add(ep)
    add(system_korean_font_path())
    if _FONT_DIR.is_dir():
        for p in sorted(_FONT_DIR.glob("*.ttf")):
            add(str(p.resolve()))
        for p in sorted(_FONT_DIR.glob("*.otf")):
            add(str(p.resolve()))
    return out


CHART_FONT = "Malgun Gothic, Apple SD Gothic Neo, Pretendard, sans-serif"


def get_korean_font_path() -> str | None:
    """UI 표시용: 번들 → 시스템 순."""
    b = bundled_wordcloud_font_path()
    if b:
        return b
    return system_korean_font_path()


CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=CHART_FONT, size=13, color="#1a1a2e"),
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,250,252,0.95)",
    margin=dict(l=24, r=24, t=56, b=24),
    hoverlabel=dict(font=dict(family=CHART_FONT)),
)

PLOTLY_CONFIG: dict = {"displayModeBar": True, "responsive": True}

DASH_METRIC_CSS = """
<style>
section.main [data-testid="stMetric"] {
    background: linear-gradient(165deg, #ffffff 0%, #f8fafc 60%, #eef2ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    border-left: 4px solid #6366f1;
}
section.main [data-testid="stMetric"] label p {
    color: #64748b !important;
    font-weight: 500 !important;
}
section.main [data-testid="stMetric"] [data-testid="stMetricValue"] > div {
    color: #0f172a !important;
}
</style>
"""

DASH_USER_ACCENTS = ("#4f46e5", "#0d9488", "#db2777", "#d97706", "#7c3aed", "#059669")


def inject_dashboard_metric_styles() -> None:
    st.markdown(DASH_METRIC_CSS, unsafe_allow_html=True)

KR_WEEKDAY_SHORT = ["월", "화", "수", "목", "금", "토", "일"]
KR_WEEKDAY_LONG = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

REQUIRED_COLUMNS = ("Date", "User", "Message")

DATE_LINE_RE = re.compile(
    r"^--------------- (?P<y>\d{4})년 (?P<m>\d{1,2})월 (?P<d>\d{1,2})일 .+ ---------------$"
)
MSG_LINE_RE = re.compile(
    r"^\[(?P<user>[^\]]+)\] \[(?P<ap>오전|오후) (?P<h>\d{1,2}):(?P<mi>\d{2})\] (?P<msg>.*)$"
)

URL_RE = re.compile(
    r"https?://[^\s<>'\"()\[\]{}]+"
    r"|www\.[^\s<>'\"()\[\]{}]+"
)

SYSTEM_MESSAGE_TOKENS = frozenset({"사진", "이모티콘", "동영상", "보이스톡"})

TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}")


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def is_numbers_only_message(text: str) -> bool:
    """숫자·시간·기호만 있는 메시지(통화시간 등)는 단어 분석에서 제외."""
    t = strip_urls(text)
    t = re.sub(r"\s+", "", t.strip())
    if not t:
        return True
    if re.search(r"[가-힣A-Za-z]", t):
        return False
    return bool(re.fullmatch(r"[0-9:.,+/%/-]+", t))


def should_exclude_message_for_word_analysis(msg: str) -> bool:
    raw = (msg or "").strip()
    if not raw:
        return True
    core = strip_urls(raw).strip()
    if not core:
        return True
    if core in SYSTEM_MESSAGE_TOKENS:
        return True
    if is_numbers_only_message(raw):
        return True
    return False


def tokenize_for_wordfreq(text: str) -> list[str]:
    t = strip_urls(text or "")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    out: list[str] = []
    for m in TOKEN_RE.finditer(t):
        w = m.group(0)
        if len(w) < 2:
            continue
        if w in SYSTEM_MESSAGE_TOKENS:
            continue
        if w.isdigit():
            continue
        out.append(w)
    return out


def global_word_counter(df: pd.DataFrame) -> Counter[str]:
    c: Counter[str] = Counter()
    for msg in df["Message"].fillna("").astype(str):
        if should_exclude_message_for_word_analysis(msg):
            continue
        c.update(tokenize_for_wordfreq(msg))
    return c


def word_counter_by_user(df: pd.DataFrame) -> dict[str, Counter[str]]:
    by_user: dict[str, Counter[str]] = {}
    sub = df[["User", "Message"]].copy()
    sub["User"] = sub["User"].fillna("").astype(str)
    for user, g in sub.groupby("User", sort=False):
        c: Counter[str] = Counter()
        for msg in g["Message"].fillna("").astype(str):
            if should_exclude_message_for_word_analysis(msg):
                continue
            c.update(tokenize_for_wordfreq(msg))
        by_user[user] = c
    return by_user


def max_consecutive_char(msg: str, ch: str) -> int:
    best = cur = 0
    for c in msg or "":
        if c == ch:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def laugh_bucket_counts(series: pd.Series, ch: str) -> dict[str, int]:
    """메시지별 최대 연속 길이로 구간을 나눈 건수(서로 배타)."""
    n0 = n1 = n2 = n3p = 0
    for msg in series.fillna("").astype(str):
        m = max_consecutive_char(msg, ch)
        if m == 0:
            n0 += 1
        elif m == 1:
            n1 += 1
        elif m == 2:
            n2 += 1
        else:
            n3p += 1
    return {
        f"{ch} 없음": n0,
        f"최대 1연속 ({ch})": n1,
        f"최대 2연속 ({ch}{ch})": n2,
        f"최대 3연속 이상 ({ch * 3}~)": n3p,
    }


def fig_top_words_bar(counter: Counter[str], top_n: int, title: str) -> go.Figure:
    most = counter.most_common(top_n)
    if not most:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=18)), height=200)
        fig.add_annotation(text="표시할 단어가 없습니다.", showarrow=False, y=0.5, x=0.5)
        return fig
    words, counts = zip(*most, strict=True)
    d = pd.DataFrame({"단어": words, "빈도": counts}).iloc[::-1]
    fig = px.bar(
        d,
        x="빈도",
        y="단어",
        orientation="h",
        color="빈도",
        color_continuous_scale=[[0, "#fef3c7"], [0.5, "#f59e0b"], [1, "#b45309"]],
        text="빈도",
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text:,}",
        textfont=dict(size=11, color="#78350f"),
        marker_line_width=0,
        hovertemplate="%{y} — %{x:,}회<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="등장 횟수",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(380, 28 * top_n + 100),
        showlegend=False,
    )
    return fig


def fig_laugh_compare(series: pd.Series, ch: str, title: str) -> go.Figure:
    buckets = laugh_bucket_counts(series, ch)
    order = list(buckets.keys())
    vals = [buckets[k] for k in order]
    fig = px.bar(x=order, y=vals, labels={"x": "구간", "y": "메시지 수"})
    fig.update_traces(
        marker_color="#8b5cf6",
        marker_line_width=0,
        hovertemplate="%{x}<br>%{y:,}건<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=None,
        yaxis_title="메시지 수",
        height=400,
        showlegend=False,
    )
    return fig


def build_wordcloud_image(freq: dict[str, int], font_path: str | None) -> io.BytesIO | None:
    if not freq:
        return None

    candidates = iter_wordcloud_font_candidates(font_path)
    base_kw: dict = dict(
        width=1100,
        height=550,
        background_color="white",
        max_words=120,
        prefer_horizontal=0.88,
        colormap="Spectral",
        min_font_size=10,
        relative_scaling=0.45,
    )

    last_err: Exception | None = None
    for fp in candidates:
        use_fp, cleanup_paths = materialize_font_for_cloud(fp)
        try:
            wc = WordCloud(font_path=use_fp, **base_kw)
            wc.generate_from_frequencies(freq)
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            buf.seek(0)
            return buf
        except (OSError, ValueError, RuntimeError) as e:
            last_err = e
            continue
        finally:
            for p in cleanup_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    try:
        wc = WordCloud(**base_kw)
        wc.generate_from_frequencies(freq)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        return buf
    except (OSError, ValueError, RuntimeError) as e:
        last_err = e

    st.warning(
        "워드클라우드용 폰트를 불러오지 못했습니다. "
        "`fonts/NanumGothic-Regular.ttf`가 **실제 TTF(수 MB)** 로 커밋됐는지, "
        f"Git LFS 포인터가 아닌지 확인하세요. ({last_err!r})"
    )
    return None


def _load_csv(file: BinaryIO) -> pd.DataFrame:
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8-sig")


def _decode_upload(file: BinaryIO) -> str:
    file.seek(0)
    raw = file.read()
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _ampm_to_hour24(ap: str, h: int) -> int:
    if ap == "오전":
        return 0 if h == 12 else h
    if h == 12:
        return 12
    return h + 12


def parse_kakaotalk_txt(text: str) -> pd.DataFrame:
    """카카오톡 모바일 스타일 TXT(날짜 구분선 + [이름] [오전/오후 시:분] 메시지) 파싱."""
    rows: list[tuple[datetime, str, str]] = []
    current_ymd: tuple[int, int, int] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        dm = DATE_LINE_RE.match(stripped)
        if dm:
            current_ymd = (int(dm["y"]), int(dm["m"]), int(dm["d"]))
            continue

        mm = MSG_LINE_RE.match(stripped)
        if mm and current_ymd is not None:
            y, mo, d = current_ymd
            h24 = _ampm_to_hour24(mm["ap"], int(mm["h"]))
            ts = datetime(y, mo, d, h24, int(mm["mi"]))
            rows.append((ts, mm["user"], mm["msg"]))
            continue

        if rows:
            ts, user, prev = rows[-1]
            rows[-1] = (ts, user, prev + "\n" + line)

    if not rows:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _participant_message_lengths(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["User", "Message"]].copy()
    out["User"] = out["User"].fillna("").astype(str)
    out["Message"] = out["Message"].fillna("").astype(str)
    out["_chars"] = out["Message"].str.len()
    return out


def participant_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    w = _participant_message_lengths(df)
    agg = (
        w.groupby("User", as_index=False)
        .agg(
            메시지_개수=("Message", "count"),
            총_글자수=("_chars", "sum"),
            평균_메시지_길이=("_chars", "mean"),
        )
        .sort_values("메시지_개수", ascending=False)
        .reset_index(drop=True)
    )
    agg["평균_메시지_길이"] = agg["평균_메시지_길이"].round(1)
    return agg


def longest_message_row(df: pd.DataFrame) -> pd.Series | None:
    w = _participant_message_lengths(df)
    if w.empty or w["_chars"].max() == 0:
        return None
    return w.loc[w["_chars"].idxmax()]


def fig_message_count_bar(stats: pd.DataFrame) -> go.Figure:
    d = stats.sort_values("메시지_개수", ascending=True)
    fig = px.bar(
        d,
        x="메시지_개수",
        y="User",
        orientation="h",
        color="메시지_개수",
        color_continuous_scale=[[0, "#e0e7ff"], [0.5, "#818cf8"], [1, "#4338ca"]],
        text="메시지_개수",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="#312e81"),
        marker_line_width=0,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="참여자별 메시지 개수", font=dict(size=18)),
        xaxis_title="메시지 수",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(360, 48 * len(d) + 120),
        showlegend=False,
    )
    return fig


def fig_total_chars_bar(stats: pd.DataFrame) -> go.Figure:
    d = stats.sort_values("총_글자수", ascending=True)
    fig = px.bar(
        d,
        x="총_글자수",
        y="User",
        orientation="h",
        color="총_글자수",
        color_continuous_scale=[[0, "#fce7f3"], [0.5, "#f472b6"], [1, "#be185d"]],
        text="총_글자수",
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text:,}",
        textfont=dict(size=12, color="#831843"),
        marker_line_width=0,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="참여자별 총 글자 수", font=dict(size=18)),
        xaxis_title="글자 수 (공백·줄바꿈 포함)",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(360, 48 * len(d) + 120),
        showlegend=False,
    )
    return fig


def fig_avg_length_bar(stats: pd.DataFrame) -> go.Figure:
    d = stats.sort_values("평균_메시지_길이", ascending=True)
    fig = px.bar(
        d,
        x="평균_메시지_길이",
        y="User",
        orientation="h",
        color="평균_메시지_길이",
        color_continuous_scale=[[0, "#ccfbf1"], [0.5, "#2dd4bf"], [1, "#0f766e"]],
        text="평균_메시지_길이",
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text:.1f}",
        textfont=dict(size=12, color="#134e4a"),
        marker_line_width=0,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="참여자별 평균 메시지 길이", font=dict(size=18)),
        xaxis_title="글자 수 (메시지당 평균)",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(320, 44 * len(d) + 100),
        showlegend=False,
    )
    return fig


def parse_dates_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def hourly_message_counts(tdf: pd.DataFrame) -> pd.Series:
    return (
        tdf["Date"]
        .dt.hour.value_counts()
        .reindex(range(24), fill_value=0)
        .astype(int)
        .rename("messages")
    )


def weekday_message_counts(tdf: pd.DataFrame) -> pd.Series:
    return (
        tdf["Date"]
        .dt.dayofweek.value_counts()
        .reindex(range(7), fill_value=0)
        .astype(int)
        .rename("messages")
    )


def monthly_message_counts(tdf: pd.DataFrame) -> pd.DataFrame:
    ser = tdf.groupby(tdf["Date"].dt.to_period("M")).size().sort_index()
    return pd.DataFrame(
        {"월_시작일": ser.index.to_timestamp(how="start"), "메시지_수": ser.values.astype(int)}
    )


def fig_hourly_messages(tdf: pd.DataFrame) -> go.Figure:
    c = hourly_message_counts(tdf)
    fig = px.bar(
        x=c.index.astype(int),
        y=c.values,
        labels={"x": "시각 (시)", "y": "메시지 수"},
    )
    fig.update_traces(
        marker_color="#6366f1",
        marker_line_width=0,
        hovertemplate="%{x}시 — %{y:,}건<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="시간대별 (0~23시) 메시지 분포", font=dict(size=18)),
        xaxis=dict(dtick=1, title="시 (24시제)", range=[-0.5, 23.5]),
        yaxis_title="메시지 수",
        bargap=0.12,
        height=420,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def fig_weekday_messages(tdf: pd.DataFrame) -> go.Figure:
    c = weekday_message_counts(tdf)
    fig = px.bar(
        x=KR_WEEKDAY_SHORT,
        y=c.values,
        labels={"x": "요일", "y": "메시지 수"},
        category_orders={"x": KR_WEEKDAY_SHORT},
    )
    fig.update_traces(
        marker_color="#ec4899",
        marker_line_width=0,
        hovertemplate="%{x} — %{y:,}건<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="요일별 메시지 분포", font=dict(size=18)),
        xaxis_title=None,
        yaxis_title="메시지 수",
        bargap=0.18,
        height=420,
        hovermode="x",
        showlegend=False,
    )
    return fig


def fig_monthly_trend(month_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        month_df,
        x="월_시작일",
        y="메시지_수",
        markers=True,
        labels={"월_시작일": "월", "메시지_수": "메시지 수"},
    )
    fig.update_traces(
        line=dict(color="#0d9488", width=2.5),
        marker=dict(size=8, color="#0f766e"),
        hovertemplate="%{x|%Y-%m} — %{y:,}건<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="월별 메시지 추이", font=dict(size=18)),
        xaxis_title=None,
        yaxis_title="메시지 수",
        height=420,
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(tickformat="%Y-%m")
    return fig


def peak_hour_and_weekday(tdf: pd.DataFrame) -> tuple[int, str]:
    hc = hourly_message_counts(tdf)
    wc = weekday_message_counts(tdf)
    peak_h = int(hc.idxmax())
    peak_dow = int(wc.idxmax())
    return peak_h, KR_WEEKDAY_LONG[peak_dow]


REPLY_GAP_NEW_SESSION_SEC = 6 * 3600
REPLY_GAP_IGNORE_SEC = 3600


def collect_reply_transitions(tdf: pd.DataFrame) -> list[dict]:
    """연속 동일 발화자 블록 사이 간격. 6시간 이상 공백은 제외. 서로 다른 사용자만."""
    if len(tdf) < 2:
        return []
    d = tdf.sort_values("Date", ascending=True).reset_index(drop=True)
    blocks: list[dict] = []
    for _, row in d.iterrows():
        u = str(row["User"])
        ts = pd.Timestamp(row["Date"])
        msg = str(row.get("Message", ""))
        if blocks and blocks[-1]["user"] == u:
            blocks[-1]["end"] = ts
            blocks[-1]["tail_message"] = msg
        else:
            blocks.append(
                {
                    "user": u,
                    "start": ts,
                    "end": ts,
                    "head_message": msg,
                    "tail_message": msg,
                }
            )

    out: list[dict] = []
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        if a["user"] == b["user"]:
            continue
        gap_sec = float((b["start"] - a["end"]).total_seconds())
        if gap_sec < 0:
            continue
        if gap_sec >= REPLY_GAP_NEW_SESSION_SEC:
            continue
        out.append(
            {
                "from_user": a["user"],
                "to_user": b["user"],
                "gap_sec": gap_sec,
                "reply_at": b["start"],
                "after_preview": (a["tail_message"] or "")[:100],
                "reply_preview": (b["head_message"] or "")[:100],
            }
        )
    return out


def format_duration_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}초"
    m, s = divmod(int(round(sec)), 60)
    if m < 60:
        return f"{m}분 {s}초"
    h, m = divmod(m, 60)
    return f"{h}시간 {m}분"


def avg_my_reply_after_others(transitions: list[dict], me: str) -> float | None:
    gaps = [t["gap_sec"] for t in transitions if t["from_user"] != me and t["to_user"] == me]
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def reply_speed_by_responder(transitions: list[dict]) -> pd.DataFrame:
    if not transitions:
        return pd.DataFrame(columns=["User", "평균_답장_초", "답장_횟수"])
    rows = []
    for t in transitions:
        rows.append({"User": t["to_user"], "gap_sec": t["gap_sec"]})
    g = pd.DataFrame(rows).groupby("User", as_index=False).agg(
        평균_답장_초=("gap_sec", "mean"),
        답장_횟수=("gap_sec", "count"),
    )
    g["평균_답장_초"] = g["평균_답장_초"].round(1)
    return g.sort_values("평균_답장_초", ascending=True).reset_index(drop=True)


def top5_fastest_my_replies(transitions: list[dict], me: str) -> pd.DataFrame:
    mine = [t for t in transitions if t["from_user"] != me and t["to_user"] == me]
    mine.sort(key=lambda x: x["gap_sec"])
    top = mine[:5]
    if not top:
        return pd.DataFrame(
            columns=[
                "순위",
                "상대",
                "답장_초",
                "답장까지",
                "답장_시각",
                "상대_마지막_말",
                "내_첫_답장",
            ]
        )
    rows = []
    for i, t in enumerate(top, start=1):
        g = t["gap_sec"]
        rows.append(
            {
                "순위": i,
                "상대": t["from_user"],
                "답장_초": round(g, 1),
                "답장까지": format_duration_seconds(g),
                "답장_시각": t["reply_at"],
                "상대_마지막_말": t["after_preview"],
                "내_첫_답장": t["reply_preview"],
            }
        )
    return pd.DataFrame(rows)


def count_slow_replies_my(transitions: list[dict], me: str) -> int:
    """상대 블록 직후 내가 답했으나 1시간 이상 걸린 횟수(6시간 미만 구간만 집계)."""
    return sum(
        1
        for t in transitions
        if t["from_user"] != me
        and t["to_user"] == me
        and t["gap_sec"] >= REPLY_GAP_IGNORE_SEC
    )


def fig_reply_speed_compare(by_user: pd.DataFrame) -> go.Figure:
    if by_user.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, title=dict(text="참여자별 평균 답장 속도", font=dict(size=18)), height=220)
        fig.add_annotation(text="비교할 답장 구간이 없습니다.", showarrow=False, x=0.5, y=0.5)
        return fig
    d = by_user.copy()
    d["평균_분"] = d["평균_답장_초"] / 60.0
    d = d.sort_values("평균_분", ascending=True)
    fig = px.bar(
        d,
        x="평균_분",
        y="User",
        orientation="h",
        color="평균_분",
        color_continuous_scale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
        text="답장_횟수",
        custom_data=["평균_답장_초", "답장_횟수"],
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="n=%{text}",
        textfont=dict(size=11, color="#1e3a8a"),
        hovertemplate=(
            "%{y}<br>평균: %{customdata[0]:.1f}초 "
            "(%{x:.2f}분)<br>샘플 수: %{customdata[1]}<extra></extra>"
        ),
        marker_line_width=0,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="참여자별 평균 답장 속도 (다른 사람 발화 블록 직후)", font=dict(size=18)),
        xaxis_title="평균 답장까지 걸린 시간 (분)",
        yaxis_title=None,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(320, 44 * len(d) + 100),
        showlegend=False,
    )
    return fig


def _load_dataframe(uploaded: BinaryIO, name: str) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith(".txt"):
        text = _decode_upload(uploaded)
        return parse_kakaotalk_txt(text)
    uploaded.seek(0)
    return _load_csv(uploaded)


def main() -> None:
    st.set_page_config(
        page_title="카카오톡 대화 분석기",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("## 💬 카카오톡 대화 분석기")
        st.caption(
            "카카오톡에서보낸 **CSV** 또는 **TXT** 대화 파일을 올리면 "
            "메시지·시간·단어·답장 속도 등을 한눈에 볼 수 있습니다."
        )
        st.markdown("---")
        uploaded = st.file_uploader(
            "CSV 또는 TXT 파일 업로드",
            type=["csv", "txt"],
            help="CSV는 Date, User, Message 열이 필요합니다. TXT는 모바일 대화보내기 형식을 권장합니다.",
        )
        st.markdown("---")
        st.markdown("### 📖 사용 방법")
        st.markdown(
            """
1. **파일 선택**  
   위에서 CSV 또는 TXT를 업로드합니다.

2. **CSV 형식**  
   `Date`, `User`, `Message` 열이 있어야 합니다. (날짜는 자동으로 해석합니다.)

3. **TXT 형식**  
   카카오톡 **모바일**에서 저장한 대화(날짜 구분선 + `[이름] [오전/오후 시:분]` 줄)를 지원합니다.

4. **탭으로 보기**  
   기본 통계 → 시간 → 단어 → 답장 속도 순으로 확인합니다.

5. **답장 속도**  
   해당 탭에서 **내 닉네임**을 골라야 내 기준 통계가 나옵니다.
            """
        )

    if uploaded is None:
        st.info("👈 **왼쪽 사이드바**에서 대화 파일을 업로드하면 분석 탭이 표시됩니다.")
        return

    name = uploaded.name or ""
    try:
        df = _load_dataframe(uploaded, name)
    except Exception as e:  # noqa: BLE001
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    if name.lower().endswith(".txt") and len(df) == 0:
        st.error(
            "TXT에서 메시지를 찾지 못했습니다. "
            "PC용 카카오톡보내기(쉼표로 구분된 한 줄 형식)는 형식이 달라 지원하지 않을 수 있습니다."
        )
        return

    missing = _validate_columns(df)
    if missing:
        st.error(
            "필수 컬럼이 없습니다: "
            + ", ".join(missing)
            + f"\n\n현재 컬럼: {', '.join(map(str, df.columns))}"
        )
        return

    df = parse_dates_column(df)
    invalid_dates = int(df["Date"].isna().sum())
    tdf = df.dropna(subset=["Date"])

    users = df["User"].dropna().astype(str).unique()
    participants = sorted(users.tolist(), key=str.lower)
    stats = participant_stats_table(df)
    total_word_counter = global_word_counter(df)
    by_user_words = word_counter_by_user(df)
    user_word_keys = sorted(by_user_words.keys(), key=str.lower)
    word_font = get_korean_font_path()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📎 업로드된 파일")
        st.text(name or "(이름 없음)")
        st.metric("메시지 수", f"{len(df):,}")
        st.metric("참여자 수", f"{len(users):,}")
        st.markdown("**참여자**")
        st.caption(" · ".join(participants) if participants else "—")
        if len(tdf) > 0:
            d_min = tdf["Date"].min()
            d_max = tdf["Date"].max()
            st.markdown("**대화 기간**")
            st.caption(
                f"{pd.Timestamp(d_min).strftime('%Y-%m-%d %H:%M')}  ~  "
                f"{pd.Timestamp(d_max).strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.warning("유효한 날짜가 없어 기간을 표시할 수 없습니다.")

    if invalid_dates:
        st.warning(
            f"`Date` 컬럼을 날짜로 해석하지 못한 행이 **{invalid_dates:,}개** 있습니다. "
            "해당 행은 시간·답장 속도 분석에서 제외됩니다."
        )

    tab_basic, tab_time, tab_words, tab_reply = st.tabs(
        ("📊 기본 통계", "⏰ 시간 분석", "💬 단어 분석", "💌 답장 속도")
    )

    with tab_basic:
        if not st.session_state.get("_dash_metric_css_loaded"):
            inject_dashboard_metric_styles()
            st.session_state["_dash_metric_css_loaded"] = True

        st.markdown(
            '<p style="font-size:1.35rem;font-weight:700;color:#1e1b4b;margin:0 0 4px 0;">'
            "📊 대화 요약"
            "</p>"
            '<p style="color:#64748b;margin:0 0 1rem 0;">주요 지표를 카드 형태로 모았습니다.</p>',
            unsafe_allow_html=True,
        )

        n_msg = len(df)
        n_user = len(users)
        total_chars = int(stats["총_글자수"].sum()) if len(stats) else 0
        overall_avg_len = (total_chars / n_msg) if n_msg else 0.0

        if len(tdf) > 0:
            d_min = pd.Timestamp(tdf["Date"].min())
            d_max = pd.Timestamp(tdf["Date"].max())
            span_days = int(
                (tdf["Date"].dt.normalize().max() - tdf["Date"].dt.normalize().min()).days + 1
            )
            daily_avg = n_msg / span_days if span_days > 0 else 0.0
            span_label = f"{d_min.strftime('%Y.%m.%d')} ~ {d_max.strftime('%Y.%m.%d')}"
        else:
            span_days = 0
            daily_avg = 0.0
            span_label = "날짜 없음"

        stats_sorted = stats.sort_values("메시지_개수", ascending=False).reset_index(drop=True)
        top_u = str(stats_sorted.iloc[0]["User"]) if len(stats_sorted) else "—"

        r1 = st.columns(4)
        with r1[0]:
            st.metric("전체 메시지", f"{n_msg:,}", help="파일에 포함된 모든 메시지 줄 수")
        with r1[1]:
            st.metric("참여자 수", f"{n_user:,}명")
        with r1[2]:
            st.metric(
                "대화 기간",
                f"{span_days:,}일" if span_days else "—",
                help="유효한 Date 기준, 달력 일 수(첫날·마지막 날 포함)",
            )
        with r1[3]:
            st.metric(
                "일평균 메시지",
                f"{daily_avg:.1f}건/일" if span_days else "—",
                help="전체 메시지 ÷ 대화 기간(일)",
            )
        if len(tdf) > 0:
            st.caption(f"📅 첫 메시지 ~ 마지막 메시지: **{span_label}**")

        r2 = st.columns(3)
        with r2[0]:
            st.metric("총 글자 수", f"{total_chars:,}", help="모든 메시지 길이 합(공백·줄바꿈 포함)")
        with r2[1]:
            st.metric("전체 평균 메시지 길이", f"{overall_avg_len:.1f}자", help="총 글자 수 ÷ 전체 메시지 수")
        with r2[2]:
            st.metric("최다 발화자", top_u, help="메시지 개수 기준 1위")

        st.markdown("---")
        st.markdown(
            '<p style="font-size:1.15rem;font-weight:700;color:#334155;margin:0 0 0.75rem 0;">'
            "👥 참여자별 카드"
            "</p>",
            unsafe_allow_html=True,
        )

        n_rows = len(stats_sorted)
        chunk = 3
        for i in range(0, n_rows, chunk):
            row_df = stats_sorted.iloc[i : i + chunk]
            cols = st.columns(len(row_df))
            for j, (_, srow) in enumerate(row_df.iterrows()):
                u = str(srow["User"])
                accent = DASH_USER_ACCENTS[(i + j) % len(DASH_USER_ACCENTS)]
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(
                            f'<p style="margin:0 0 10px 0;padding:8px 10px;border-radius:10px;'
                            f"background:linear-gradient(90deg,{accent}18,{accent}08);"
                            f'color:{accent};font-weight:700;font-size:1.05rem;">'
                            f"👤 {html.escape(u)}"
                            "</p>",
                            unsafe_allow_html=True,
                        )
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("메시지", f"{int(srow['메시지_개수']):,}")
                        with c2:
                            st.metric("총 글자", f"{int(srow['총_글자수']):,}")
                        with c3:
                            st.metric("평균 길이", f"{float(srow['평균_메시지_길이']):.1f}자")

        st.markdown("---")
        st.markdown("**차트**")
        c_a, c_b = st.columns(2)
        with c_a:
            st.plotly_chart(
                fig_message_count_bar(stats), use_container_width=True, config=PLOTLY_CONFIG
            )
        with c_b:
            st.plotly_chart(
                fig_avg_length_bar(stats), use_container_width=True, config=PLOTLY_CONFIG
            )

        with st.expander("📋 참여자별 수치 표", expanded=False):
            show_tbl = stats.rename(
                columns={
                    "메시지_개수": "메시지 개수",
                    "총_글자수": "총 글자 수",
                    "평균_메시지_길이": "평균 메시지 길이",
                }
            )
            st.dataframe(
                show_tbl.style.format(
                    {
                        "메시지 개수": "{:,}",
                        "총 글자 수": "{:,}",
                        "평균 메시지 길이": "{:.1f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🔍 데이터 미리보기 (10행)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 💬 가장 긴 메시지")
        longest = longest_message_row(df)
        if longest is None:
            st.caption("표시할 메시지가 없습니다.")
        else:
            n_chars = int(longest["_chars"])
            st.markdown(
                f"**{longest['User']}** · {n_chars:,}자 "
                "(공백·줄바꿈 포함 길이 기준)"
            )
            with st.container(border=True):
                st.text(longest["Message"])

    with tab_time:
        if len(tdf) == 0:
            st.info("유효한 `Date` 값이 없어 시간대·요일·월별 차트를 표시할 수 없습니다.")
        else:
            m1, m2 = st.columns(2)
            peak_h, peak_day_long = peak_hour_and_weekday(tdf)
            with m1:
                st.metric("가장 대화가 활발한 시간대", f"{peak_h}시 (24시제)")
            with m2:
                st.metric("가장 대화가 활발한 요일", peak_day_long)

            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    fig_hourly_messages(tdf), use_container_width=True, config=PLOTLY_CONFIG
                )
            with ch2:
                st.plotly_chart(
                    fig_weekday_messages(tdf), use_container_width=True, config=PLOTLY_CONFIG
                )

            month_df = monthly_message_counts(tdf)
            if not month_df.empty:
                st.plotly_chart(
                    fig_monthly_trend(month_df), use_container_width=True, config=PLOTLY_CONFIG
                )

    with tab_words:
        st.caption(
            "제외: 1글자 단어, 시스템 메시지(사진·이모티콘·동영상·보이스톡)만 있는 메시지·해당 토큰, "
            "URL, 숫자·기호만 있는 메시지. 한글·영문 연속 2글자 이상을 단어로 집계합니다."
        )
        if word_font:
            st.caption(f"워드클라우드 폰트: `{word_font}`")
        else:
            st.warning(
                "프로젝트 `fonts/NanumGothic-Regular.ttf` 또는 시스템 한글 폰트를 찾지 못했습니다. "
                "나눔고딕 TTF를 `fonts/`에 넣고 다시 배포하세요."
            )

        with st.expander("워드클라우드 한글이 네모일 때 (폰트 점검)", expanded=False):
            p = _PRIMARY_BUNDLED_FONT
            st.markdown(
                f"- **앱 기준 경로:** `{_APP_DIR}`\n"
                f"- **번들 폰트 파일:** `{p}`\n"
                f"- **존재:** `{p.is_file()}`\n"
            )
            if p.is_file():
                sz = p.stat().st_size
                st.markdown(
                    f"- **크기:** `{sz:,}` bytes "
                    f"(Git LFS 포인터는 보통 100~200바이트입니다. 실제 TTF는 **1MB 이상**이어야 합니다.)\n"
                    f"- **PIL 로드:** `{_pil_truetype_ok(str(p.resolve()))}`\n"
                    f"- **한글 음절 cmap:** `{_font_cmap_contains_hangul(str(p.resolve()))}`\n"
                )
            cands = iter_wordcloud_font_candidates(word_font)
            st.markdown(f"- **실제 사용 후보 순서:** `{cands}`")

        st.plotly_chart(
            fig_top_words_bar(total_word_counter, 20, "전체 — 가장 많이 쓴 단어 TOP 20"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        if user_word_keys:
            sel_user = st.selectbox(
                "참여자 선택 (TOP 10 단어)", user_word_keys, key="wordfreq_user"
            )
            st.plotly_chart(
                fig_top_words_bar(
                    by_user_words[sel_user],
                    10,
                    f"`{sel_user}` — 가장 많이 쓴 단어 TOP 10",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        st.markdown("##### ☁️ 워드클라우드")
        cloud_freq = dict(total_word_counter.most_common(200))
        wc_buf = build_wordcloud_image(cloud_freq, word_font)
        if wc_buf is not None:
            st.image(wc_buf, use_container_width=True)
        elif cloud_freq:
            st.caption(
                "워드클라우드 이미지를 만들지 못했습니다. "
                "`fonts/NanumGothic-Regular.ttf`가 앱과 같은 폴더 구조로 배포됐는지 확인하세요."
            )
        else:
            st.caption("워드클라우드를 그릴 만한 단어가 없습니다.")

        st.markdown("##### 😆 ㅋ / 😊 ㅎ 연속 길이별 메시지 수")
        st.caption("각 메시지에서 **가장 긴 연속 ㅋ(또는 ㅎ)** 기준으로 하나의 구간에만 집계합니다.")
        lx1, lx2 = st.columns(2)
        with lx1:
            st.plotly_chart(
                fig_laugh_compare(
                    df["Message"],
                    "ㅋ",
                    "메시지당 최대 연속 ㅋ 길이 분포",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with lx2:
            st.plotly_chart(
                fig_laugh_compare(
                    df["Message"],
                    "ㅎ",
                    "메시지당 최대 연속 ㅎ 길이 분포",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    with tab_reply:
        st.caption(
            "연속으로 같은 사람이 보낸 메시지는 **한 블록**으로 묶고, "
            "블록이 바뀔 때 **이전 블록 마지막 시각 → 다음 블록 첫 시각** 간격을 답장 지연으로 봅니다. "
            "이전 블록과 다음 블록 사이가 **6시간 이상**이면 새 대화로 보고 **제외**합니다."
        )
        if len(tdf) < 2:
            st.info("유효한 `Date`가 2개 미만이면 답장 속도를 계산할 수 없습니다.")
        else:
            transitions = collect_reply_transitions(tdf)
            me = st.selectbox(
                "내 닉네임 (상대 이후 내 답장·TOP5·읽씹 기준)",
                options=participants,
                key="reply_speed_me",
            )
            my_avg = avg_my_reply_after_others(transitions, me)
            slow_cnt = count_slow_replies_my(transitions, me)
            m_a, m_b, m_c = st.columns(3)
            with m_a:
                if my_avg is None:
                    st.metric(
                        "⏱️ 상대 발화 후 내 답장 평균",
                        "—",
                        help="다른 사람 블록 직후, 내가 이어서 보낸 블록만 집계",
                    )
                else:
                    st.metric(
                        "⏱️ 상대 발화 후 내 답장 평균",
                        format_duration_seconds(my_avg),
                        help="다른 사람 블록 직후, 내가 이어서 보낸 블록만 집계",
                    )
            with m_b:
                st.metric(
                    "🐢 1시간 이상 뒤에 한 답장(내 기준)",
                    f"{slow_cnt:,}",
                    help="상대 블록 직후 내 블록까지 1시간 이상, 6시간 미만인 경우",
                )
            with m_c:
                st.metric(
                    "📎 분석에 쓴 답장 구간 수(전체)",
                    f"{len(transitions):,}",
                    help="6시간 미만 간격·발화자 교차만",
                )

            by_reply = reply_speed_by_responder(transitions)
            st.plotly_chart(
                fig_reply_speed_compare(by_reply),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            st.markdown("##### ⚡ 가장 빠른 답장 TOP 5 (내 기준)")
            top5 = top5_fastest_my_replies(transitions, me)
            if top5.empty:
                st.caption("표시할 내 답장 구간이 없습니다.")
            else:
                show5 = top5.copy()
                show5["답장_시각"] = pd.to_datetime(show5["답장_시각"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
                st.dataframe(
                    show5[
                        [
                            "순위",
                            "상대",
                            "답장_초",
                            "답장까지",
                            "답장_시각",
                            "상대_마지막_말",
                            "내_첫_답장",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("답장까지 걸린 시간은 **초 단위**로 계산한 뒤 읽기 좋게 표시했습니다.")


if __name__ == "__main__":
    main()
