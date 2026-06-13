import os
import platform
import re
from collections import Counter
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

REQUIRED_COLUMNS = ["Date", "User", "Message"]
CHART_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
DASHBOARD_CARD_COLORS = ["#636EFA", "#00CC96", "#EF553B", "#AB63FA", "#FFA15A", "#19D3F3"]
DAY_NAMES_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
SYSTEM_MESSAGES = {"사진", "이모티콘", "동영상", "보이스톡"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[가-힣a-zA-Z0-9]+")
NUMBERS_ONLY_PATTERN = re.compile(r"^[\d\s:.,\-+분초시간전후초]+$")
NEW_CONVERSATION_GAP = pd.Timedelta(hours=6)
READ_IGNORE_THRESHOLD = pd.Timedelta(hours=1)


def _unique_existing_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    existing: list[str] = []
    for path in paths:
        normalized = str(Path(path).resolve())
        if normalized in seen or not os.path.isfile(normalized):
            continue
        seen.add(normalized)
        existing.append(normalized)
    return existing


def get_korean_font_candidates() -> list[str]:
    bundled_name = "assets/fonts/NotoSansKR-Regular.otf"
    candidates: list[str] = []

    for root in (_PROJECT_DIR, Path.cwd()):
        candidates.append(str((root / bundled_name).resolve()))

    system = platform.system()
    if system == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates.extend(
            [
                os.path.join(windir, "Fonts", "malgun.ttf"),
                os.path.join(windir, "Fonts", "malgunbd.ttf"),
            ]
        )
    elif system == "Darwin":
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/Library/Fonts/AppleGothic.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            ]
        )

    preferred_names = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    import matplotlib.font_manager as fm

    for name in preferred_names:
        try:
            candidates.append(fm.findfont(name, fallback_to_default=False))
        except (ValueError, OSError):
            continue

    for font in fm.fontManager.ttflist:
        if any(keyword in font.name for keyword in ("Gothic", "Nanum", "Malgun", "Noto Sans KR")):
            if "Hiragino" not in font.name:
                candidates.append(font.fname)

    return _unique_existing_paths(candidates)


def build_wordcloud(word_frequencies: dict[str, int]) -> WordCloud:
    last_error: Exception | None = None
    for font_path in get_korean_font_candidates():
        try:
            return WordCloud(
                font_path=font_path,
                width=900,
                height=450,
                background_color="white",
                colormap="Set2",
                max_words=80,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_frequencies)
        except (OSError, ValueError) as exc:
            last_error = exc
            continue

    message = "한글 폰트를 찾지 못해 워드클라우드를 생성할 수 없습니다."
    if last_error is not None:
        message = f"{message} ({last_error})"
    raise RuntimeError(message)


def should_exclude_message(message: str) -> bool:
    text = message.strip()
    if not text:
        return True
    if text in SYSTEM_MESSAGES:
        return True
    if URL_PATTERN.search(text):
        return True
    if NUMBERS_ONLY_PATTERN.fullmatch(text):
        return True
    return False


def extract_words(message: str) -> list[str]:
    cleaned = URL_PATTERN.sub(" ", message)
    words = []
    for token in TOKEN_PATTERN.findall(cleaned):
        if len(token) < 2:
            continue
        if token in SYSTEM_MESSAGES:
            continue
        if token.isdigit():
            continue
        words.append(token)
    return words


def count_words(messages: pd.Series) -> Counter:
    counter: Counter = Counter()
    for message in messages:
        if should_exclude_message(message):
            continue
        counter.update(extract_words(message))
    return counter


def top_words_dataframe(counter: Counter, limit: int) -> pd.DataFrame:
    items = counter.most_common(limit)
    return pd.DataFrame(items, columns=["단어", "횟수"])


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}초"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초" if secs else f"{minutes}분"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}시간 {minutes}분" if minutes else f"{hours}시간"


def build_conversation_turns(df: pd.DataFrame) -> pd.DataFrame:
    """연속으로 보낸 같은 사람 메시지를 하나의 턴으로 묶습니다."""
    sorted_df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if sorted_df.empty:
        return pd.DataFrame(
            columns=["User", "StartDate", "EndDate", "FirstMessage", "LastMessage"]
        )

    turns: list[dict] = []
    current_user = sorted_df.iloc[0]["User"]
    start_idx = 0

    for idx in range(1, len(sorted_df)):
        if sorted_df.iloc[idx]["User"] != current_user:
            block = sorted_df.iloc[start_idx:idx]
            turns.append(
                {
                    "User": current_user,
                    "StartDate": block.iloc[0]["Date"],
                    "EndDate": block.iloc[-1]["Date"],
                    "FirstMessage": block.iloc[0]["Message"],
                    "LastMessage": block.iloc[-1]["Message"],
                }
            )
            current_user = sorted_df.iloc[idx]["User"]
            start_idx = idx

    block = sorted_df.iloc[start_idx:]
    turns.append(
        {
            "User": current_user,
            "StartDate": block.iloc[0]["Date"],
            "EndDate": block.iloc[-1]["Date"],
            "FirstMessage": block.iloc[0]["Message"],
            "LastMessage": block.iloc[-1]["Message"],
        }
    )
    return pd.DataFrame(turns)


def compute_reply_events(turns: pd.DataFrame) -> pd.DataFrame:
    """턴 사이 답장 이벤트를 계산합니다. 6시간 이상 공백은 새 대화로 제외합니다."""
    if len(turns) < 2:
        return pd.DataFrame(
            columns=[
                "Replier",
                "RepliedTo",
                "ReplySeconds",
                "ReceivedAt",
                "RepliedAt",
                "ReceivedMessage",
                "ReplyMessage",
            ]
        )

    events: list[dict] = []
    for idx in range(len(turns) - 1):
        prev_turn = turns.iloc[idx]
        next_turn = turns.iloc[idx + 1]
        gap = next_turn["StartDate"] - prev_turn["EndDate"]
        if gap >= NEW_CONVERSATION_GAP:
            continue

        reply_seconds = gap.total_seconds()
        events.append(
            {
                "Replier": next_turn["User"],
                "RepliedTo": prev_turn["User"],
                "ReplySeconds": reply_seconds,
                "ReceivedAt": prev_turn["EndDate"],
                "RepliedAt": next_turn["StartDate"],
                "ReceivedMessage": prev_turn["LastMessage"],
                "ReplyMessage": next_turn["FirstMessage"],
            }
        )
    return pd.DataFrame(events)


def analyze_reply_speed(reply_events: pd.DataFrame, me: str) -> dict:
    my_replies = reply_events[reply_events["Replier"] == me]
    read_ignore_count = int((my_replies["ReplySeconds"] >= READ_IGNORE_THRESHOLD.total_seconds()).sum())

    participant_stats = (
        reply_events.groupby("Replier", as_index=False)
        .agg(
            평균_답장_초=("ReplySeconds", "mean"),
            답장_횟수=("ReplySeconds", "count"),
            읽씹_횟수=("ReplySeconds", lambda s: (s >= READ_IGNORE_THRESHOLD.total_seconds()).sum()),
        )
        .sort_values("평균_답장_초")
    )
    participant_stats["평균_답장_시간"] = participant_stats["평균_답장_초"].apply(format_duration)

    fastest = my_replies.nsmallest(5, "ReplySeconds").copy()
    if not fastest.empty:
        fastest["답장_시간"] = fastest["ReplySeconds"].apply(format_duration)

    my_avg_seconds = my_replies["ReplySeconds"].mean() if not my_replies.empty else None

    return {
        "my_avg_seconds": my_avg_seconds,
        "my_reply_count": len(my_replies),
        "read_ignore_count": read_ignore_count,
        "participant_stats": participant_stats,
        "fastest_replies": fastest,
        "reply_events": reply_events,
    }


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: transparent;
        }
        .dashboard-card {
            border-radius: 12px;
            padding: 0.25rem 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
        }
        .dashboard-card-primary { border-left: 4px solid #636EFA; background: linear-gradient(135deg, #636EFA14 0%, #636EFA06 100%); }
        .dashboard-card-success { border-left: 4px solid #00CC96; background: linear-gradient(135deg, #00CC9614 0%, #00CC9606 100%); }
        .dashboard-card-warning { border-left: 4px solid #FFA15A; background: linear-gradient(135deg, #FFA15A14 0%, #FFA15A06 100%); }
        .dashboard-card-danger  { border-left: 4px solid #EF553B; background: linear-gradient(135deg, #EF553B14 0%, #EF553B06 100%); }
        .dashboard-card-purple  { border-left: 4px solid #AB63FA; background: linear-gradient(135deg, #AB63FA14 0%, #AB63FA06 100%); }
        .dashboard-card-cyan    { border-left: 4px solid #19D3F3; background: linear-gradient(135deg, #19D3F314 0%, #19D3F306 100%); }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #AB63FA !important;
        }
        .stTabs button[data-baseweb="tab"]:hover,
        .stTabs button[data-baseweb="tab"]:hover > div[data-testid="stMarkdownContainer"] > p {
            color: #AB63FA !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    card_class: str = "dashboard-card-primary",
) -> None:
    delta_arg = delta if delta else None
    st.markdown(f'<div class="dashboard-card {card_class}">', unsafe_allow_html=True)
    st.metric(label, value, delta_arg)
    st.markdown("</div>", unsafe_allow_html=True)


def style_chart(fig, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        template="plotly_white",
        font=dict(family="Apple SD Gothic Neo, Malgun Gothic, sans-serif", size=13),
        margin=dict(t=70, b=40, l=40, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig

def load_chat_dataframe(uploaded_file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp949")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.sidebar.error(
            f"필수 칼럼이 없습니다: {', '.join(missing_columns)}. "
            "Date, User, Message 칼럼이 포함된 CSV를 업로드해 주세요."
        )
        return None

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["User"] = df["User"].astype(str).str.strip()
    df["Message"] = df["Message"].astype(str)
    df["MessageLength"] = df["Message"].str.len()
    return df


st.set_page_config(page_title="카카오톡 대화 분석기", page_icon="💬", layout="wide")

with st.sidebar:
    st.title("💬 카카오톡 대화 분석기")
    st.caption(
        "카카오톡 대화보내기 CSV를 업로드하면 "
        "메시지 통계, 시간 패턴, 단어 분석, 답장 속도를 한눈에 볼 수 있습니다."
    )

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

    df: pd.DataFrame | None = None
    if uploaded_file is not None:
        df = load_chat_dataframe(uploaded_file)
        if df is not None:
            participants_preview = sorted(df["User"].dropna().unique())
            valid_dates = df["Date"].dropna()

            st.divider()
            st.subheader("📋 기본 정보")
            st.metric("메시지 수", f"{len(df):,}개")
            st.markdown("**참여자**")
            st.write(", ".join(participants_preview))
            if not valid_dates.empty:
                start_date = valid_dates.min().strftime("%Y-%m-%d")
                end_date = valid_dates.max().strftime("%Y-%m-%d")
                st.markdown("**기간**")
                st.write(f"{start_date} ~ {end_date}")
            else:
                st.caption("날짜 정보를 파싱할 수 없어 기간을 표시하지 못했습니다.")

    st.divider()
    st.subheader("📖 사용 방법")
    st.markdown(
        """
        1. 카카오톡에서 대화방 **설정 → 대화 내용보내기**로 CSV 파일을 저장합니다.
        2. 왼쪽 **CSV 파일 업로드** 버튼으로 파일을 선택합니다.
        3. 메인 화면의 탭에서 분석 결과를 확인합니다.
           - **기본 통계**: 참여자별 메시지·글자 수
           - **시간 분석**: 시간대·요일·월별 활동
           - **단어 분석**: 자주 쓰는 단어, 워드클라우드
           - **답장 속도**: 평균 답장 시간, 읽씹 횟수
        """
    )

if uploaded_file is None or df is None:
    st.stop()

user_stats = (
    df.groupby("User", as_index=False)
    .agg(
        메시지_개수=("Message", "count"),
        총_글자_수=("MessageLength", "sum"),
        평균_메시지_길이=("MessageLength", "mean"),
    )
    .sort_values("메시지_개수", ascending=False)
)
longest_idx = df["MessageLength"].idxmax()
longest_row = df.loc[longest_idx]
participants = sorted(df["User"].dropna().unique())

inject_dashboard_styles()

tab_stats, tab_time, tab_words, tab_reply = st.tabs(
    ["📊 기본 통계", "🕐 시간 분석", "📝 단어 분석", "⏱️ 답장 속도"]
)

with tab_stats:
    valid_dates = df["Date"].dropna()
    total_messages = len(df)
    total_chars = int(df["MessageLength"].sum())
    avg_msg_length = df["MessageLength"].mean()

    if not valid_dates.empty:
        start_date = valid_dates.min()
        end_date = valid_dates.max()
        duration_days = max((end_date - start_date).days + 1, 1)
        daily_avg = total_messages / duration_days
        date_range_label = f"{start_date.strftime('%Y.%m.%d')} ~ {end_date.strftime('%Y.%m.%d')}"
        duration_label = f"{duration_days:,}일"
    else:
        duration_days = None
        daily_avg = None
        date_range_label = "—"
        duration_label = "—"

    st.markdown("### 📊 대화 개요")
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1:
        metric_card("전체 메시지", f"{total_messages:,}개", card_class="dashboard-card-primary")
    with overview_col2:
        metric_card(
            "참여자 수",
            f"{len(participants)}명",
            ", ".join(participants),
            card_class="dashboard-card-success",
        )
    with overview_col3:
        metric_card(
            "대화 기간",
            duration_label,
            date_range_label if not valid_dates.empty else None,
            card_class="dashboard-card-warning",
        )
    with overview_col4:
        daily_label = f"{daily_avg:.1f}개" if daily_avg is not None else "—"
        metric_card(
            "일평균 메시지",
            daily_label,
            "활동일 기준" if daily_avg is not None else None,
            card_class="dashboard-card-purple",
        )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        metric_card("총 글자 수", f"{total_chars:,}자", card_class="dashboard-card-cyan")
    with summary_col2:
        metric_card("평균 메시지 길이", f"{avg_msg_length:.1f}자", card_class="dashboard-card-danger")
    with summary_col3:
        msg_share_top = user_stats.iloc[0]
        share_pct = msg_share_top["메시지_개수"] / total_messages * 100
        metric_card(
            "가장 많이 보낸 사람",
            msg_share_top["User"],
            f"{int(msg_share_top['메시지_개수']):,}개 ({share_pct:.1f}%)",
            card_class="dashboard-card-primary",
        )

    st.divider()
    st.markdown("### 👥 참여자별 통계")

    participant_cols = st.columns(min(len(user_stats), 3))
    for card_idx, (_, row) in enumerate(user_stats.iterrows()):
        color = DASHBOARD_CARD_COLORS[card_idx % len(DASHBOARD_CARD_COLORS)]
        msg_pct = row["메시지_개수"] / total_messages * 100
        with participant_cols[card_idx % len(participant_cols)]:
            with st.container(border=True):
                st.markdown(
                    f'<p style="color:{color}; font-size:1.05rem; font-weight:700; margin:0 0 0.5rem;">'
                    f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                    f'background:{color};margin-right:6px;"></span>{row["User"]}</p>',
                    unsafe_allow_html=True,
                )
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("메시지", f"{int(row['메시지_개수']):,}개", f"{msg_pct:.1f}%")
                with p_col2:
                    st.metric("총 글자", f"{int(row['총_글자_수']):,}자")
                st.metric("평균 길이", f"{row['평균_메시지_길이']:.1f}자")

    st.divider()
    st.markdown("### 📈 시각화")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        msg_chart = px.bar(
            user_stats.sort_values("메시지_개수"),
            x="메시지_개수",
            y="User",
            orientation="h",
            color="User",
            color_discrete_sequence=CHART_COLORS,
            text="메시지_개수",
            labels={"메시지_개수": "메시지 개수", "User": "참여자"},
        )
        msg_chart.update_traces(texttemplate="%{text:,}개", textposition="outside")
        style_chart(msg_chart, "참여자별 메시지 개수")
        msg_chart.update_layout(showlegend=False, height=320)
        st.plotly_chart(msg_chart, use_container_width=True)

    with chart_col2:
        char_chart = px.bar(
            user_stats.sort_values("총_글자_수"),
            x="총_글자_수",
            y="User",
            orientation="h",
            color="User",
            color_discrete_sequence=CHART_COLORS[::-1],
            text="총_글자_수",
            labels={"총_글자_수": "총 글자 수", "User": "참여자"},
        )
        char_chart.update_traces(texttemplate="%{text:,}자", textposition="outside")
        style_chart(char_chart, "참여자별 총 글자 수")
        char_chart.update_layout(showlegend=False, height=320)
        st.plotly_chart(char_chart, use_container_width=True)

    avg_chart = px.bar(
        user_stats.sort_values("평균_메시지_길이"),
        x="평균_메시지_길이",
        y="User",
        orientation="h",
        color="User",
        color_discrete_sequence=CHART_COLORS,
        text="평균_메시지_길이",
        labels={"평균_메시지_길이": "평균 메시지 길이 (자)", "User": "참여자"},
    )
    avg_chart.update_traces(
        texttemplate="%{text:.1f}자",
        textposition="outside",
    )
    style_chart(avg_chart, "참여자별 평균 메시지 길이")
    avg_chart.update_layout(showlegend=False, height=280)
    st.plotly_chart(avg_chart, use_container_width=True)

    display_stats = user_stats.copy()
    display_stats["평균_메시지_길이"] = display_stats["평균_메시지_길이"].round(1)
    display_stats.columns = ["참여자", "메시지 개수", "총 글자 수", "평균 메시지 길이"]
    st.dataframe(display_stats, use_container_width=True, hide_index=True)

    st.subheader("🏆 가장 긴 메시지")
    longest_col1, longest_col2 = st.columns([1, 3])
    with longest_col1:
        st.metric("보낸 사람", longest_row["User"])
        st.metric("글자 수", f"{longest_row['MessageLength']:,}자")
    with longest_col2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
                border-left: 4px solid #636EFA;
                border-radius: 8px;
                padding: 1rem 1.25rem;
                font-size: 1rem;
                line-height: 1.6;
                word-break: break-word;
            ">
                {longest_row["Message"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pd.notna(longest_row["Date"]):
            st.caption(f"전송 시각: {longest_row['Date'].strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("👀 데이터 미리보기 (처음 10개 행)")
    st.dataframe(df[REQUIRED_COLUMNS].head(10), use_container_width=True, hide_index=True)

with tab_time:
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates:
        st.warning(f"Date를 파싱하지 못한 메시지 {invalid_dates:,}개는 시간 분석에서 제외됩니다.")

    df_time = df.dropna(subset=["Date"]).copy()
    df_time["Hour"] = df_time["Date"].dt.hour
    df_time["DayOfWeek"] = df_time["Date"].dt.dayofweek
    df_time["DayName"] = df_time["DayOfWeek"].map(dict(enumerate(DAY_NAMES_KO)))
    df_time["YearMonth"] = df_time["Date"].dt.to_period("M").astype(str)

    hour_stats = (
        df_time.groupby("Hour")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index(name="메시지_개수")
    )
    hour_stats["시간대"] = hour_stats["Hour"].apply(lambda h: f"{h}시")

    day_stats = (
        df_time.groupby("DayOfWeek")
        .size()
        .reindex(range(7), fill_value=0)
        .reset_index(name="메시지_개수")
    )
    day_stats["DayName"] = day_stats["DayOfWeek"].map(dict(enumerate(DAY_NAMES_KO)))

    month_stats = (
        df_time.groupby("YearMonth")
        .size()
        .reset_index(name="메시지_개수")
        .sort_values("YearMonth")
    )

    peak_hour_row = hour_stats.loc[hour_stats["메시지_개수"].idxmax()]
    peak_day_row = day_stats.loc[day_stats["메시지_개수"].idxmax()]

    time_metric_col1, time_metric_col2 = st.columns(2)
    with time_metric_col1:
        st.metric(
            "가장 활발한 시간대",
            f"{int(peak_hour_row['Hour'])}시",
            f"{int(peak_hour_row['메시지_개수']):,}개 메시지",
        )
    with time_metric_col2:
        st.metric(
            "가장 활발한 요일",
            peak_day_row["DayName"],
            f"{int(peak_day_row['메시지_개수']):,}개 메시지",
        )

    time_chart_col1, time_chart_col2 = st.columns(2)

    with time_chart_col1:
        hour_colors = [
            CHART_COLORS[0] if h == peak_hour_row["Hour"] else "rgba(99, 110, 250, 0.45)"
            for h in hour_stats["Hour"]
        ]
        hour_chart = go.Figure(
            go.Bar(
                x=hour_stats["시간대"],
                y=hour_stats["메시지_개수"],
                marker_color=hour_colors,
                text=hour_stats["메시지_개수"],
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate="%{x}<br>메시지 %{y:,}개<extra></extra>",
            )
        )
        style_chart(hour_chart, "시간대별 메시지 분포 (0시~23시)")
        hour_chart.update_layout(
            height=360,
            xaxis=dict(categoryorder="array", categoryarray=hour_stats["시간대"].tolist()),
            yaxis_title="메시지 개수",
        )
        st.plotly_chart(hour_chart, use_container_width=True)

    with time_chart_col2:
        day_colors = [
            CHART_COLORS[1] if d == peak_day_row["DayOfWeek"] else "rgba(239, 85, 59, 0.45)"
            for d in day_stats["DayOfWeek"]
        ]
        day_chart = go.Figure(
            go.Bar(
                x=day_stats["DayName"],
                y=day_stats["메시지_개수"],
                marker_color=day_colors,
                text=day_stats["메시지_개수"],
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate="%{x}<br>메시지 %{y:,}개<extra></extra>",
            )
        )
        style_chart(day_chart, "요일별 메시지 분포")
        day_chart.update_layout(
            height=360,
            xaxis=dict(categoryorder="array", categoryarray=DAY_NAMES_KO),
            yaxis_title="메시지 개수",
        )
        st.plotly_chart(day_chart, use_container_width=True)

    month_chart = go.Figure(
        go.Scatter(
            x=month_stats["YearMonth"],
            y=month_stats["메시지_개수"],
            mode="lines+markers",
            line=dict(color=CHART_COLORS[2], width=3),
            marker=dict(size=9, color=CHART_COLORS[2], line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.12)",
            hovertemplate="%{x}<br>메시지 %{y:,}개<extra></extra>",
        )
    )
    style_chart(month_chart, "월별 메시지 추이")
    month_chart.update_layout(height=340, xaxis_title="월", yaxis_title="메시지 개수")
    st.plotly_chart(month_chart, use_container_width=True)

with tab_words:
    analyzable_df = df[~df["Message"].apply(should_exclude_message)].copy()
    excluded_count = len(df) - len(analyzable_df)
    if excluded_count:
        st.caption(
            f"시스템 메시지, URL, 숫자만 있는 메시지 등 {excluded_count:,}개는 단어 분석에서 제외했습니다."
        )

    all_word_counts = count_words(analyzable_df["Message"])
    top20 = top_words_dataframe(all_word_counts, 20)

    if top20.empty:
        st.info("분석할 수 있는 단어가 없습니다.")
    else:
        top20_chart = px.bar(
            top20.sort_values("횟수"),
            x="횟수",
            y="단어",
            orientation="h",
            color="횟수",
            color_continuous_scale=[[0, "rgba(99, 110, 250, 0.5)"], [1, CHART_COLORS[0]]],
            text="횟수",
            labels={"횟수": "사용 횟수", "단어": "단어"},
        )
        top20_chart.update_traces(texttemplate="%{text}회", textposition="outside")
        style_chart(top20_chart, "전체 메시지 자주 쓰는 단어 TOP 20")
        top20_chart.update_layout(showlegend=False, height=560, coloraxis_showscale=False)
        st.plotly_chart(top20_chart, use_container_width=True)

        st.markdown("#### 참여자별 자주 쓰는 단어 TOP 10")
        participant_cols = st.columns(min(len(participants), 3))
        for idx, user in enumerate(participants):
            user_words = count_words(analyzable_df.loc[analyzable_df["User"] == user, "Message"])
            user_top10 = top_words_dataframe(user_words, 10)
            with participant_cols[idx % len(participant_cols)]:
                if user_top10.empty:
                    st.write(f"**{user}**")
                    st.caption("분석할 단어가 없습니다.")
                    continue
                user_chart = px.bar(
                    user_top10.sort_values("횟수"),
                    x="횟수",
                    y="단어",
                    orientation="h",
                    color_discrete_sequence=[CHART_COLORS[idx % len(CHART_COLORS)]],
                    text="횟수",
                    labels={"횟수": "사용 횟수", "단어": "단어"},
                )
                user_chart.update_traces(texttemplate="%{text}회", textposition="outside")
                style_chart(user_chart, user)
                user_chart.update_layout(showlegend=False, height=360)
                st.plotly_chart(user_chart, use_container_width=True)

        st.markdown("#### ☁️ 워드클라우드")
        try:
            with st.spinner("워드클라우드를 생성하는 중..."):
                wc = build_wordcloud(dict(all_word_counts.most_common(80)))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.patch.set_facecolor("white")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except RuntimeError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {exc}")

    st.markdown("#### ㅋ / ㅎ 포함 메시지 비교")
    laugh_df = df.copy()
    laugh_df["has_k"] = laugh_df["Message"].str.contains("ㅋ", regex=False)
    laugh_df["has_h"] = laugh_df["Message"].str.contains("ㅎ", regex=False)

    laugh_stats = (
        laugh_df.groupby("User", as_index=False)
        .agg(
            k_count=("has_k", "sum"),
            h_count=("has_h", "sum"),
            total_count=("Message", "count"),
        )
        .sort_values("k_count", ascending=False)
    )

    laugh_col1, laugh_col2 = st.columns(2)

    with laugh_col1:
        k_chart = px.bar(
            laugh_stats.sort_values("k_count"),
            x="k_count",
            y="User",
            orientation="h",
            color="User",
            color_discrete_sequence=CHART_COLORS,
            text="k_count",
            labels={"k_count": "메시지 수", "User": "참여자"},
        )
        k_chart.update_traces(texttemplate="%{text}개", textposition="outside")
        style_chart(k_chart, '"ㅋ" 포함 메시지 (ㅋ, ㅋㅋ, ㅋㅋㅋ 등)')
        k_chart.update_layout(showlegend=False, height=320)
        st.plotly_chart(k_chart, use_container_width=True)

    with laugh_col2:
        h_chart = px.bar(
            laugh_stats.sort_values("h_count"),
            x="h_count",
            y="User",
            orientation="h",
            color="User",
            color_discrete_sequence=CHART_COLORS[::-1],
            text="h_count",
            labels={"h_count": "메시지 수", "User": "참여자"},
        )
        h_chart.update_traces(texttemplate="%{text}개", textposition="outside")
        style_chart(h_chart, '"ㅎ" 포함 메시지 (ㅎ, ㅎㅎ, ㅎㅎㅎ 등)')
        h_chart.update_layout(showlegend=False, height=320)
        st.plotly_chart(h_chart, use_container_width=True)

    laugh_display = laugh_stats.rename(
        columns={
            "User": "참여자",
            "k_count": "ㅋ 포함 메시지",
            "h_count": "ㅎ 포함 메시지",
            "total_count": "전체 메시지",
        }
    )
    laugh_display["ㅋ 비율 (%)"] = (
        laugh_display["ㅋ 포함 메시지"] / laugh_display["전체 메시지"] * 100
    ).round(1)
    laugh_display["ㅎ 비율 (%)"] = (
        laugh_display["ㅎ 포함 메시지"] / laugh_display["전체 메시지"] * 100
    ).round(1)
    st.dataframe(laugh_display, use_container_width=True, hide_index=True)

with tab_reply:
    df_reply = df.dropna(subset=["Date"]).copy()
    conversation_turns = build_conversation_turns(df_reply)
    reply_events = compute_reply_events(conversation_turns)

    me = st.selectbox(
        "나는 누구인가요?",
        options=participants,
        help="상대방 메시지 이후 내 답장 속도를 계산할 참여자를 선택하세요.",
    )

    if reply_events.empty:
        st.info("답장 속도를 계산할 수 있는 대화 턴이 충분하지 않습니다.")
    else:
        reply_analysis = analyze_reply_speed(reply_events, me)

        reply_metric_col1, reply_metric_col2, reply_metric_col3 = st.columns(3)
        with reply_metric_col1:
            if reply_analysis["my_avg_seconds"] is not None:
                st.metric(
                    "내 평균 답장 속도",
                    format_duration(reply_analysis["my_avg_seconds"]),
                    f"{reply_analysis['my_reply_count']:,}회 답장",
                )
            else:
                st.metric("내 평균 답장 속도", "—", "답장 기록 없음")
        with reply_metric_col2:
            st.metric("내 읽씹 횟수", f"{reply_analysis['read_ignore_count']:,}회", "1시간 이상 미답장")
        with reply_metric_col3:
            st.metric("분석된 답장 쌍", f"{len(reply_events):,}개", "6시간 공백 제외")

        st.caption(
            "같은 사람이 연속으로 보낸 메시지는 하나의 턴으로 묶었고, "
            "6시간 이상 대화 공백은 새 대화로 보아 답장 시간 계산에서 제외했습니다."
        )

        reply_chart_col1, reply_chart_col2 = st.columns(2)

        participant_stats = reply_analysis["participant_stats"]
        with reply_chart_col1:
            speed_chart = px.bar(
                participant_stats.sort_values("평균_답장_초", ascending=False),
                x="평균_답장_초",
                y="Replier",
                orientation="h",
                color="Replier",
                color_discrete_sequence=CHART_COLORS,
                text="평균_답장_시간",
                labels={"평균_답장_초": "평균 답장 시간 (초)", "Replier": "참여자"},
            )
            speed_chart.update_traces(textposition="outside")
            style_chart(speed_chart, "참여자별 평균 답장 속도")
            speed_chart.update_layout(showlegend=False, height=320)
            st.plotly_chart(speed_chart, use_container_width=True)

        with reply_chart_col2:
            ignore_chart = px.bar(
                participant_stats.sort_values("읽씹_횟수"),
                x="읽씹_횟수",
                y="Replier",
                orientation="h",
                color="Replier",
                color_discrete_sequence=CHART_COLORS[::-1],
                text="읽씹_횟수",
                labels={"읽씹_횟수": "읽씹 횟수", "Replier": "참여자"},
            )
            ignore_chart.update_traces(texttemplate="%{text}회", textposition="outside")
            style_chart(ignore_chart, "참여자별 읽씹 횟수 (1시간 이상)")
            ignore_chart.update_layout(showlegend=False, height=320)
            st.plotly_chart(ignore_chart, use_container_width=True)

        participant_display = participant_stats.rename(
            columns={
                "Replier": "참여자",
                "답장_횟수": "답장 횟수",
                "읽씹_횟수": "읽씹 횟수",
            }
        )[["참여자", "평균_답장_시간", "답장 횟수", "읽씹 횟수"]]
        st.dataframe(participant_display, use_container_width=True, hide_index=True)

        fastest = reply_analysis["fastest_replies"]
        st.markdown(f"#### 🚀 {me}의 가장 빠른 답장 TOP 5")
        if fastest.empty:
            st.caption("답장 기록이 없습니다.")
        else:
            fastest_display = fastest[
                ["RepliedTo", "ReplySeconds", "ReceivedAt", "ReceivedMessage", "ReplyMessage"]
            ].copy()
            fastest_display["ReceivedAt"] = fastest_display["ReceivedAt"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            fastest_display["ReplySeconds"] = fastest_display["ReplySeconds"].apply(
                lambda s: f"{s:.0f}초"
            )
            fastest_display.columns = [
                "상대방",
                "답장 속도",
                "받은 시각",
                "받은 메시지",
                "내 답장",
            ]
            st.dataframe(fastest_display, use_container_width=True, hide_index=True)
