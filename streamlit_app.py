"""
אפליקציית Streamlit לחיזוי "חוזק קבוצה" בפרמייר ליג
------------------------------------------------------

מה האפליקציה עושה?
- טוענת את נתוני השחקנים (אותו CSV שבו משתמש הסקריפט `premier_league_team_strength_model.py`).
- בונה פיצ'רים ברמת קבוצה ומאמנת מודל כמו בסקריפט המקורי.
- מאפשרת לבחור שתי קבוצות מתפריטי בחירה ולחשב לכל אחת:
  הסתברות להיות מוגדרת כ"קבוצה חזקה" (label = 1).
- הקבוצה עם הסתברות גבוהה יותר נחשבת כפייבוריט התיאורטי (מי "תנצח").

איך מריצים את האפליקציה?
1. ודא שהקובץ `premier_league_team_strength_model.py` וקובץ ה-CSV נמצאים באותה תיקייה/מבנה,
   כמו בסקריפט המקורי (ברירת המחדל היא: `data/premier_league_players.csv`).
2. התקן חבילות נדרשות (פעם אחת):
   pip install streamlit scikit-learn pandas numpy
3. מתוך תיקיית הפרויקט (שבה נמצא קובץ זה), הרץ:
   streamlit run streamlit_app.py
4. הדפדפן ייפתח ותוכל לבחור שתי קבוצות ולקבל תחזית.
"""

import time
from pathlib import Path
import difflib

import plotly.graph_objects as go

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from api_data_fetcher import fetch_premier_league_standings_df

# מייבאים פונקציות ולוגיקה מהסקריפט הקיים
from premier_league_team_strength_model import (
    DEFAULT_CSV_PATH,
    load_player_data,
    preprocess_players,
    build_team_level_features,
    train_model,
)


# ==========================
# פונקציות עזר ל-RTL ו-UI
# ==========================

def inject_global_css() -> None:
    """
    מזריק לחומרת האפליקציה CSS גלובלי.
    כולל הגדרות לכיווניות RTL מימין לשמאל, עיצוב 'כרטיסיות' (Cards),
    ופורמט ממורכז (Centered text).
    """
    st.markdown(
        """
        <style>
        /* כיווניות גלובלית לימין */
        .block-container {
            direction: rtl;
            text-align: right;
        }
        
        /* מחלקת כרטיסייה (Card) שנעטוף בה אזורים מרכזיים */
        .st-card {
            background-color: rgba(255, 255, 255, 0.05); /* מותאם ל-Dark Mode */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ffffff; /* טקסט לבן ל-Dark Mode */
        }

        /* מחלקות ייעודיות לטקסט ממורכז */
        .text-center {
            text-align: center !important;
        }
        
        .score-board-team {
            font-size: 1.5rem;
            font-weight: bold;
            color: white !important; /* כדי לבלוט ב-Dark Mode */
        }
        
        .score-board-prob {
            font-size: 3rem;
            font-weight: 900;
            color: #1e88e5; /* ניתן לשנות את הצבע דינמית אח"כ */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def rtl(text: str) -> None:
    """
    מציג טקסט בעברית עם כיווניות ימין-לשמאל באמצעות st.markdown ו-HTML מפורש.
    (שמור לתאימות לאחור, למרות שיש הגדרה גלובלית עכשיו)
    """
    st.markdown(
        f'<div dir="rtl" style="text-align: right;">{text}</div>',
        unsafe_allow_html=True,
    )

def render_colored_form_badge(form_str: str) -> str:
    """
    מקבל מחרוזת כמו 'WWDLW' ומחזיר HTML מעוצב עם חתיכות בצבעים נפרדים:
    W = ירוק, D = אפור/כתום, L = אדום.
    """
    if not form_str:
        return "N/A"
        
    html_parts = []
    for char in form_str.upper():
        if char == 'W':
            bg_color = "#4caf50" # ירוק
        elif char == 'D':
            bg_color = "#9e9e9e" # אפור
        elif char == 'L':
            bg_color = "#f44336" # אדום
        else:
            continue
            
        html_parts.append(
            f"<span style='display:inline-block; background-color:{bg_color}; "
            f"width:32px; height:32px; line-height:32px; text-align:center; "
            f"border-radius:4px; font-weight:bold; margin: 0 2px; color:white !important; "
            f"font-size: 0.85em;'>{char}</span>"
        )
    return "".join(html_parts)

def rtl_sidebar(text: str) -> None:
    """
    מציג טקסט בעברית בסרגל הצד עם כיווניות ימין-לשמאל.
    """
    st.sidebar.markdown(
        f'<div dir="rtl" style="text-align: right;">{text}</div>',
        unsafe_allow_html=True,
    )


def compute_match_outcome_probs(
    proba_home_strong: float,
    proba_away_strong: float,
    home_advantage: float = 0.05,
):
    """
    פונקציה היוריסטית לחישוב הסתברויות תוצאה למשחק:
    ניצחון בית (Home), תיקו (Draw), ניצחון חוץ (Away).

    קלט:
    - proba_home_strong: ההסתברות של המודל שהקבוצה הביתית "חזקה".
    - proba_away_strong: ההסתברות של המודל שהקבוצה האורחת "חזקה".
    - home_advantage: יתרון ביתיות (נוסף לכוח של הקבוצה הביתית).

    לוגיקה:
    1. מחשבים "חוזק אפקטיבי" לכל קבוצה:
       strength_home = proba_home_strong + home_advantage
       strength_away = proba_away_strong
    2. קובעים הסתברות לתיקו כתלות בפער הכוחות:
       - gap = abs(proba_home_strong - proba_away_strong)
       - draw_base גבוה כשהפער קטן, נמוך כשהפער גדול.
    3. את השארית (1 - draw_prob) מחלקים בין Home/Away
       באופן יחסי לחוזקים האפקטיביים.

    התוצאה:
    שלישיית הסתברויות (home_win, draw, away_win) שסכומן ~= 1.
    """

    # חוזק בסיסי (עם בונוס ביתיות לקבוצה 1)
    strength_home = np.clip(proba_home_strong + home_advantage, 0.0, 1.0)
    strength_away = np.clip(proba_away_strong, 0.0, 1.0)

    # אם מסיבה כלשהי שני החוזקים אפסיים, נחזיר 1/3-1/3-1/3
    if strength_home == 0 and strength_away == 0:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3

    # פער החוזק המקורי (בלי ביתיות) – משפיע על הסיכוי לתיקו
    gap = abs(proba_home_strong - proba_away_strong)

    # הגדרה היוריסטית: כש-gap קטן → Draw גבוה; כש-gap גדול → Draw נמוך
    max_draw = 0.45  # תיקו כמעט מחצית מהמקרים כשקבוצות כמעט שוות
    min_draw = 0.10  # מינימום תיקו כשיש פער גדול מאוד
    draw_prob = max_draw - (max_draw - min_draw) * gap
    draw_prob = float(np.clip(draw_prob, min_draw, max_draw))

    # את השארית מחלקים לפי יחסי הכוחות
    remaining = max(0.0, 1.0 - draw_prob)
    total_strength = strength_home + strength_away

    home_win_prob = remaining * (strength_home / total_strength)
    away_win_prob = remaining * (strength_away / total_strength)

    # נוודא שסכום ההסתברויות קרוב ל-1 (תיקון קטן אם צריך)
    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

    return float(home_win_prob), float(draw_prob), float(away_win_prob)


# ==========================
# נתוני ליגה חיים מה-API
# ==========================


import re

def normalize_team_name(name: str) -> str:
    """
    מנרמל שם קבוצה כדי לצמצם בעיות התאמה בין שמות מה-CSV לשמות מה-API.

    פעולות:
    - המרה לאותיות קטנות.
    - החלפת מקפים ונקודות ברווחים.
    - הסרת 'fc' / 'afc'.
    - המרת 'utd' ל-'united'.
    """

    if not isinstance(name, str):
        return ""

    s = name.lower()
    for ch in ["-", "–", "_", ".", "&"]:
        s = s.replace(ch, " ")
        
    s = re.sub(r'\butd\b', 'united', s)
    s = re.sub(r'\b(?:fc|afc)\b', '', s)
    
    s = " ".join(s.split())
    return s

TEAM_MAPPING_TO_CSV = {
    "nott'm forest": "nottingham forest",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "man united": "manchester united",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "brighton": "brighton and hove albion",
}

def get_csv_team_name(live_name: str, csv_clubs: list) -> str | None:
    """
    מנסה למצוא את השם התואם של הקבוצה מהטבלה החיה במודל ההיסטורי (CSV).
    """
    if not live_name:
        return None
        
    live_name_clean = live_name.strip()
    norm_live = normalize_team_name(live_name_clean)
    
    # 1. התאמה מדויקת
    for c in csv_clubs:
        if c.strip() == live_name_clean:
            return c
            
    # 2. התאמה מנורמלת
    for c in csv_clubs:
        if normalize_team_name(c) == norm_live:
            return c
            
    # 3. התאמה לפי מילון ידני
    if norm_live in TEAM_MAPPING_TO_CSV:
        mapped_norm = TEAM_MAPPING_TO_CSV[norm_live]
        for c in csv_clubs:
            if normalize_team_name(c) == mapped_norm:
                return c
            
    # 4. התאמה רכה חכמה
    matches = difflib.get_close_matches(norm_live, [normalize_team_name(c) for c in csv_clubs], n=1, cutoff=0.55)
    if matches:
        best_match = matches[0]
        for c in csv_clubs:
            if normalize_team_name(c) == best_match:
                return c
                
    return None


@st.cache_data
def get_live_standings_df():
    """
    מושך את טבלת הפרמייר ליג העדכנית מ-API-Football ומחזיר DataFrame.
    מוסיף גם עמודת שם מנורמל לצורך התאמות שמות (team_name_norm).
    """

    df = fetch_premier_league_standings_df()
    if "team_name" not in df.columns:
        raise ValueError("עמודת 'team_name' לא נמצאה בתוצאת ה-API.")

    df = df.copy()
    df["team_name_norm"] = df["team_name"].apply(normalize_team_name)
    return df


def find_team_in_standings(live_df: pd.DataFrame, team_name: str) -> pd.Series | None:
    """
    מחפש קבוצה מטבלת ה-API לפי שם הקבוצה מה-CSV, עם התאמה "רכה":
    - קודם כל לפי התאמה מדויקת על שם מנורמל.
    - אם אין, נשתמש ב-difflib.get_close_matches על רשימת השמות המנורמלים.
    """

    if live_df is None or live_df.empty:
        return None

    target = normalize_team_name(team_name)
    if not target:
        return None

    # התאמה ישירה
    exact_matches = live_df[live_df["team_name_norm"] == target]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    # התאמה "רכה" באמצעות difflib
    candidates = live_df["team_name_norm"].tolist()
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        fuzzy_matches = live_df[live_df["team_name_norm"] == best]
        if not fuzzy_matches.empty:
            return fuzzy_matches.iloc[0]

    return None


# ==========================
# פונקציות עזר עם cache
# ==========================

@st.cache_data
def prepare_team_data(csv_path_str: str):
    """
    טוען את הנתונים הגולמיים, מנקה אותם ובונה פיצ'רים ברמת קבוצה.

    שלבים:
    1. קריאת קובץ ה-CSV מהדיסק.
    2. ניקוי נתוני השחקנים (אותה לוגיקה כמו בסקריפט המקורי).
    3. אגרגציה לרמת קבוצה (Club) + חישוב win_rate לכל קבוצה.
    4. חישוב תגית y (קבוצה חזקה / לא) לפי חציון win_rate.
    5. בניית מטריצת פיצ'רים X (ללא העמודה win_rate).

    שימוש ב-@st.cache_data:
    - מונע טענת נתונים יקרה בכל שינוי קטן ב-UI.
    - כל עוד נתיב הקובץ לא השתנה, התוצאה תגיע מה-cache וזה מהיר יותר.
    """

    csv_path = Path(csv_path_str)
    df_players_raw = load_player_data(csv_path)
    df_players_clean = preprocess_players(df_players_raw)
    team_features = build_team_level_features(df_players_clean)

    # חישוב label "קבוצה חזקה" לפי חציון win_rate (כמו בסקריפט המקורי)
    median_win_rate = team_features["win_rate"].median()
    y = (team_features["win_rate"] >= median_win_rate).astype(int)

    # בניית X – כל הפיצ'רים המספריים חוץ מה-label
    feature_cols = [c for c in team_features.columns if c != "win_rate"]
    X = team_features[feature_cols].fillna(0.0)

    # רשימת הקבוצות לצורך התפריטים הנפתחים
    clubs = team_features.index.tolist()

    return X, y, clubs, feature_cols, team_features, df_players_clean


@st.cache_resource
def train_cached_model(csv_path_str: str):
    """
    מאמן את המודל פעם אחת ושומר אותו ב-cache של Streamlit.

    למה cache_resource?
    - אימון מודל לוקח זמן יחסי, ואין צורך לאמן מחדש בכל רענון UI.
    - ברגע שהמודל אומן פעם אחת לנתיב נתונים מסוים, אפשר להשתמש בו
      שוב ושוב בקריאות predict בלי המתנה מיותרת.
    """

    X, y, clubs, feature_cols, team_features, df_players_clean = prepare_team_data(csv_path_str)

    # חלוקה ל-train/test כדי לשמור את אותם עקרונות נגד Overfitting
    # משתמשים באותו יחס כמו בסקריפט האימון הראשי (40% ל-test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    # חישוב דיוק בסיסי על סט הבדיקה לצורך הצגה למשתמש
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, X, y, clubs, feature_cols, team_features, acc, df_players_clean


# ==========================
# הגדרת ממשק המשתמש ב-Streamlit
# ==========================

st.set_page_config(
    page_title="Premier League Team Strength Predictor",
    page_icon="⚽",
    layout="centered",
)

# הפעלת עיצוב ה-CSS שיצרנו
inject_global_css()

rtl("<h1>⚽ חיזוי חוזק קבוצות בפרמייר ליג</h1>")
rtl(
    "אפליקציה קטנה שמבוססת על נתוני השחקנים וסטטיסטיקות היסטוריות.<br>"
    "המודל מנבא לכל קבוצה הסתברות להיות מוגדרת כקבוצה חזקה, "
    "ועל בסיס זה מעריך מי הפייבוריט התיאורטי במשחק ביניהן."
)

try:
    # מאמנים את המודל (או טוענים מה-cache)
    model, X_all, y_all, clubs, feature_cols, team_features, test_accuracy, df_players_clean = train_cached_model(
        str(DEFAULT_CSV_PATH)
    )
except FileNotFoundError as e:
    rtl(
        "לא הצלחתי למצוא את קובץ הנתונים.<br>"
        f"{e}<br>"
        "עדכן את הנתיב בצד שמאל לקובץ CSV הנכון (כפי שהורדת מ-Kaggle)."
    )
    st.stop()
except Exception as e:
    rtl(f"קרתה שגיאה בזמן טעינת הנתונים או אימון המודל: {e}")
    st.stop()

# הצגת מידע כללי על המודל
rtl_sidebar("<h4>מידע על המודל</h4>")
rtl_sidebar(f"דיוק על סט הבדיקה (test accuracy): <b>{test_accuracy:.2%}</b>")
rtl_sidebar(f"מספר קבוצות בדאטה: <b>{len(clubs)}</b>")

# טבלת פרמייר ליג חיה בסיידבר
rtl_sidebar("<h4>טבלת פרמייר ליג – זמן אמת</h4>")
live_standings_df = None
try:
    live_standings_df = get_live_standings_df()
    # בוחרים רק את העמודות החשובות להצגה (בלי form כדי למנוע גלילה אופקית)
    cols_to_show = ["rank", "team_name", "played", "points"]
    existing_cols = [c for c in cols_to_show if c in live_standings_df.columns]
    sidebar_table = live_standings_df[existing_cols]
    st.sidebar.dataframe(
        sidebar_table,
        width="stretch",
        hide_index=True,  # הסתרת אינדקס ה-DataFrame
    )
except Exception as e:
    rtl_sidebar(
        f"לא הצלחתי לטעון את טבלת הליגה החיה מה-API.<br>"
        f"פרטים טכניים: {e}"
    )

# st.sidebar.markdown("<br>", unsafe_allow_html=True)
# admin_debug_mode = st.sidebar.checkbox("Admin Debug Mode", value=False)
admin_debug_mode = False  # Hidden for security

rtl("<h3>בחר שתי קבוצות להשוואה</h3>")

if len(clubs) < 2:
    rtl("נדרשות לפחות שתי קבוצות בדאטה כדי לבצע השוואה.")
    st.stop()

# 1. סינון ה-Dropdown: נציג רק קבוצות מהטבלה הלייב
if live_standings_df is not None and not live_standings_df.empty:
    options_list = sorted(live_standings_df["team_name"].dropna().unique().tolist())
else:
    options_list = sorted(clubs)

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("קבוצה 1", options_list, index=0)

with col2:
    default_index = 1 if len(options_list) > 1 else 0
    team_b = st.selectbox("קבוצה 2", options_list, index=default_index)

if team_a == team_b:
    rtl("בחר שתי קבוצות שונות כדי לבצע השוואה.")
    st.stop()

if st.button("חשב הסתברות לכל קבוצה"):
    
    csv_team_a = get_csv_team_name(team_a, clubs)
    csv_team_b = get_csv_team_name(team_b, clubs)
    
    def get_team_features_with_fallback(csv_name, original_name):
        used_fallback = False
        if csv_name and csv_name in X_all.index:
            features = X_all.loc[[csv_name]]
        else:
            # Missing Data Fallback (League Average)
            st.warning(f"⚠️ אין מספיק דאטה היסטורי עבור '{original_name}' (ייתכן שעלתה ליגה). משתמש בערכי 'ממוצע ליגה'.")
            features = X_all.mean(axis=0).to_frame().T
            used_fallback = True
        return features, used_fallback

    X_team_a, fallback_a = get_team_features_with_fallback(csv_team_a, team_a)
    X_team_b, fallback_b = get_team_features_with_fallback(csv_team_b, team_b)

    # predict_proba מחזיר הסתברות לכל מחלקה; מחלקה 1 היא "חזקה"
    proba_a_raw = float(model.predict_proba(X_team_a)[0][1])
    proba_b_raw = float(model.predict_proba(X_team_b)[0][1])
    
    # Apply a 15% penalty to the historical probability if the team relied on fallback
    proba_a = proba_a_raw * 0.85 if fallback_a else proba_a_raw
    proba_b = proba_b_raw * 0.85 if fallback_b else proba_b_raw
    
    # ------------------
    # Live Data Weighting
    # ------------------
    home_form_share = 0.5
    away_form_share = 0.5
    
    ppg_a_raw = 1.0
    ppg_b_raw = 1.0
    streak_mult_a = 1.0
    streak_mult_b = 1.0
    ppg_a_adj = 1.0
    ppg_b_adj = 1.0
    ppg_a_final = 1.0
    ppg_b_final = 1.0
    
    home_bonus_applied_to_a = 1.10 # 10% boost to home PPG to give a minor advantage
    
    if live_standings_df is not None and not live_standings_df.empty:
        row_a = find_team_in_standings(live_standings_df, team_a)
        row_b = find_team_in_standings(live_standings_df, team_b)
        
        def calc_streak_ppg(form_str):
            if not form_str or pd.isna(form_str): return 1.0
            clean_form = "".join([char for char in str(form_str).upper() if char in ['W', 'D', 'L']])[-5:]
            if not clean_form: return 1.0
            pts = sum({'W': 3, 'D': 1, 'L': 0}.get(c, 0) for c in clean_form)
            return pts / len(clean_form)
        
        if row_a is not None and row_a.get("played", 0) > 0:
            ppg_a_raw = float(row_a["points"]) / float(row_a["played"])
            streak_mult_a = calc_streak_ppg(row_a.get("form"))
            ppg_a_adj = (ppg_a_raw * 0.8) + (streak_mult_a * 0.2)
            
        if row_b is not None and row_b.get("played", 0) > 0:
            ppg_b_raw = float(row_b["points"]) / float(row_b["played"])
            streak_mult_b = calc_streak_ppg(row_b.get("form"))
            ppg_b_adj = (ppg_b_raw * 0.8) + (streak_mult_b * 0.2)
            
        # Apply Home Advantage Bonus to Adjusted PPG directly
        ppg_a_final = ppg_a_adj * home_bonus_applied_to_a
        ppg_b_final = ppg_b_adj
            
        # Power Law for current form (adjusted to 2.6 to properly reflect top-tier team gaps)
        power_a = max(ppg_a_final, 0.1) ** 2.6
        power_b = max(ppg_b_final, 0.1) ** 2.6
        home_form_share = power_a / (power_a + power_b)
        away_form_share = power_b / (power_a + power_b)
        
    # Blending Strengths (10% Historical / 90% Live Form) BEFORE computing outcome breakdown
    strength_a = (0.10 * proba_a) + (0.90 * home_form_share)
    strength_b = (0.10 * proba_b) + (0.90 * away_form_share)
        
    # חישוב הסתברויות לתוצאת משחק (Home / Draw / Away) על בסיס הכוח המשוקלל
    home_win_prob, draw_prob, away_win_prob = compute_match_outcome_probs(
        proba_home_strong=strength_a,
        proba_away_strong=strength_b,
        home_advantage=0.05,
    )
    
    # Text values for the debug expander
    hw_prob_hist = proba_a # keeping variable name to pass correctly to markdown
    aw_prob_hist = proba_b

    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    rtl("<h3 class='text-center'>תוצאות התחזית (Scoreboard)</h3>")
    
    # אזור שני העמודות בלוח התוצאות
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        st.markdown(f"<div class='text-center score-board-team'>{team_a}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='text-center score-board-prob' style='color: #1e88e5;'>{home_win_prob:.1%}</div>", unsafe_allow_html=True)
        
    with score_col2:
        st.markdown(f"<div class='text-center score-board-team'>{team_b}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='text-center score-board-prob' style='color: #43a047;'>{away_win_prob:.1%}</div>", unsafe_allow_html=True)

    # סיכום מילולי לתחזית
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

    rtl(
        f"הנחה לחישוב: {team_a} היא הקבוצה הביתית (Home), "
        f"{team_b} היא הקבוצה האורחת (Away)."
    )

    col_home, col_draw, col_away = st.columns(3)

    with col_home:
        st.metric(
            label="ניצחון קבוצה ביתית (Home)",
            value=f"{home_win_prob * 100:.1f}%",
        )

    with col_draw:
        st.metric(
            label="תיקו (Draw)",
            value=f"{draw_prob * 100:.1f}%",
        )

    with col_away:
        st.metric(
            label="ניצחון קבוצה אורחת (Away)",
            value=f"{away_win_prob * 100:.1f}%",
        )

    # קביעה מי "פייבוריט" לפי הסתברות גבוהה יותר
    eps = 0.02  # טולרנס קטן בשביל הבדלים זניחים בסיכויי ניצחון
    max_prob = max(home_win_prob, draw_prob, away_win_prob)
    
    if max_prob == draw_prob:
        rtl("<p class='text-center'>לפי המודל, התוצאה הסבירה ביותר במשחק זה היא <b>תיקו</b>.</p>")
    elif abs(home_win_prob - away_win_prob) < eps:
        rtl(
            "<p class='text-center'>לפי המודל, שתי הקבוצות כמעט שוות בחוזק שלהן – "
            "קשה להגיד מי פייבוריט מובהק.</p>"
        )
    elif home_win_prob > away_win_prob:
        rtl(
            f"<p class='text-center'>לפי המודל, <b>{team_a}</b> היא הפייבוריט התיאורטית לניצחון במשחק הזה.</p>"
        )
    else:
        rtl(
            f"<p class='text-center'>לפי המודל, <b>{team_b}</b> היא הפייבוריט התיאורטית לניצחון במשחק הזה.</p>"
        )

    rtl(
        "<p style='font-size: 0.85em; color: #666;' class='text-center'>"
        "התחזית מבוססת על מודל היברידי המשקלל נתונים היסטוריים, יתרון בית/חוץ ואת הכושר הנוכחי של הקבוצות בליגה. עם זאת, המודל הסטטיסטי אינו לוקח בחשבון אירועים נקודתיים כמו פציעות, היעדרויות שחקנים או החלטות שיפוט."
        "</p>"
    )
    
    if admin_debug_mode:
        with st.expander("דוח דיבוג אלגוריתם - מאחורי הקלעים", expanded=True):
            st.markdown(f"**{team_a} (Home)**")
            st.markdown(f"- **PPG מקורי מהטבלה:** {ppg_a_raw:.3f} | **Streak PPG (5 אחרונים):** {streak_mult_a:.3f}")
            st.markdown(f"- **PPG משוקלל (80% מקורי + 20% מומנטום):** {ppg_a_adj:.3f}")
            st.markdown(f"- **PPG סופי (אחרי בונוס ביתיות +10%):** {ppg_a_final:.3f}")
            st.markdown(f"- **הסתברות וירטואלית (Live Form Share):** {home_form_share:.1%}")
            st.markdown(f"- **הסתברות גולמית מהמודל ההיסטורי:** {proba_a:.1%} (אחרי פנלטי אם הופעל)")
            st.markdown(f"- **כוח משוקלל סופי (Strength A):** {strength_a:.1%}")
            st.markdown(f"- **פנלטי הופעל:** {fallback_a}")
            
            st.markdown("---")
            
            st.markdown(f"**{team_b} (Away)**")
            st.markdown(f"- **PPG מקורי מהטבלה:** {ppg_b_raw:.3f} | **Streak PPG (5 אחרונים):** {streak_mult_b:.3f}")
            st.markdown(f"- **PPG משוקלל (80% מקורי + 20% מומנטום):** {ppg_b_adj:.3f}")
            st.markdown(f"- **PPG סופי (ללא בונוס ביתיות):** {ppg_b_final:.3f}")
            st.markdown(f"- **הסתברות וירטואלית (Live Form Share):** {away_form_share:.1%}")
            st.markdown(f"- **הסתברות גולמית מהמודל ההיסטורי:** {proba_b:.1%} (אחרי פנלטי אם הופעל)")
            st.markdown(f"- **כוח משוקלל סופי (Strength B):** {strength_b:.1%}")
            st.markdown(f"- **פנלטי הופעל:** {fallback_b}")
            
            st.markdown("---")
            
            st.markdown("**נוסחת השילוב (Blend Strength):**")
            st.code("Strength = (0.10 * Historical) + (0.90 * Live_Form_Share)\nCompute_Match_Probs(Strength_A, Strength_B)", language="python")

        
    # סגירת ה-Card הראשון
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================
    # הקשר בזמן אמת מה-API
    # ==========================
    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    rtl("<h3 class='text-center'>הקשר בזמן אמת (Real-time Context)</h3>")
    if live_standings_df is not None:
        row_home = find_team_in_standings(live_standings_df, team_a)
        row_away = find_team_in_standings(live_standings_df, team_b)

        col_ctx_home, col_ctx_away = st.columns(2)

        with col_ctx_home:
            if row_home is not None:
                rank_home = row_home.get("rank", "?")
                raw_form_home = row_home.get("form")

                if raw_form_home:
                    form_home_clean = "".join([char for char in str(raw_form_home).upper() if char in ['W', 'D', 'L']])[-5:]
                    colored_badges = render_colored_form_badge(form_home_clean)
                    
                    rtl(
                        f"<strong>{team_a}</strong><br>"
                        f"מיקום בטבלה: <strong>{rank_home}</strong><br>"
                        f"רצף אחרון: <div dir='ltr' style='display:inline-block;'>{colored_badges}</div>"
                    )
                else:
                    rtl(
                        f"<strong>{team_a}</strong><br>"
                        f"מיקום בטבלה: {rank_home}<br>"
                        f"רצף אחרון: N/A"
                    )
            else:
                rtl(
                    f"לא נמצאו נתוני ליגה חיים מתאימים עבור {team_a} (ייתכן הבדל בשם בין ה-CSV ל-API)."
                )

        with col_ctx_away:
            if row_away is not None:
                rank_away = row_away.get("rank", "?")
                raw_form_away = row_away.get("form")

                if raw_form_away:
                    form_away_clean = "".join([char for char in str(raw_form_away).upper() if char in ['W', 'D', 'L']])[-5:]
                    colored_badges = render_colored_form_badge(form_away_clean)
                    
                    rtl(
                        f"<strong>{team_b}</strong><br>"
                        f"מיקום בטבלה: <strong>{rank_away}</strong><br>"
                        f"רצף אחרון: <div dir='ltr' style='display:inline-block;'>{colored_badges}</div>"
                    )
                else:
                    rtl(
                        f"<strong>{team_b}</strong><br>"
                        f"מיקום בטבלה: {rank_away}<br>"
                        f"רצף אחרון: N/A"
                    )
            else:
                rtl(
                    f"לא נמצאו נתוני ליגה חיים מתאימים עבור {team_b} (ייתכן הבדל בשם בין ה-CSV ל-API)."
                )
    else:
        rtl("לא ניתן להציג הקשר בזמן אמת כיוון שטבלת ה-API לא נטענה בהצלחה.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================
    # גרף השוואה חי (Head-to-Head)
    # ==========================
    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    rtl("<h3 class='text-center'>השוואת סטטיסטיקות העונה (Live Data)</h3>")
    
    if live_standings_df is not None:
        row_a = find_team_in_standings(live_standings_df, team_a)
        row_b = find_team_in_standings(live_standings_df, team_b)
        
        if row_a is not None and row_b is not None:
            # חילוץ נתונים להשוואה
            categories = ["נקודות (Points)", "שערי זכות (Goals For)", "שערי חובה (Goals Against)"]
            
            vals_a = [
                row_a.get("points", 0),
                row_a.get("goals_for", 0),
                row_a.get("goals_against", 0)
            ]
            
            vals_b = [
                row_b.get("points", 0),
                row_b.get("goals_for", 0),
                row_b.get("goals_against", 0)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=categories,
                y=vals_a,
                name=team_a,
                marker_color="#1e88e5"
            ))
            fig.add_trace(go.Bar(
                x=categories,
                y=vals_b,
                name=team_b,
                marker_color="#43a047"
            ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="קטגוריות",
                yaxis_title="ערך",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            rtl("לא נמצאו מספיק נתונים חיים (Live Data) להצגת הגרף עבור שתי הקבוצות.")
    else:
        rtl("הנתונים החיים לא נטענו, לא ניתן להציג את הגרף.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
st.divider()

# ------------------
# סימולטור דטרמיניסטי לטבלת סוף העונה
# ------------------
st.markdown('<div class="st-card">', unsafe_allow_html=True)
rtl("<h3 class='text-center'>חזה את טבלת סוף העונה (מבוסס על משחקים עתידיים) 🏆</h3>")

if st.button("חשב טבלה סופית 🏆", use_container_width=True):
    if live_standings_df is None or live_standings_df.empty:
        rtl("לא ניתן לדמות, נתוני הטבלה אינם זמינים.")
    else:
        with st.spinner("מושך משחקים עתידיים ומחשב..."):
            from api_data_fetcher import fetch_remaining_fixtures, fetch_played_matches_current_season
            try:
                fixtures = fetch_remaining_fixtures()
                played_h2h = fetch_played_matches_current_season()
                
                # מעתיק את הנקודות הנוכחיות ומנהל מעקב אחרי כמות משחקים
                points_sim = {row['team_name_norm']: row['points'] for _, row in live_standings_df.iterrows()}
                simulated_games = {row['team_name_norm']: 0 for _, row in live_standings_df.iterrows()}
                played_games = {row['team_name_norm']: row.get('played', 0) for _, row in live_standings_df.iterrows()}
                
                # חישוב כושר נוכחי (PPG Weighting)
                ppg_dict = {}
                for _, row in live_standings_df.iterrows():
                    played = row.get('played', 0)
                    pts = row['points']
                    if played > 0:
                        ppg_dict[row['team_name_norm']] = pts / played
                    else:
                        ppg_dict[row['team_name_norm']] = 1.0 # ערך דיפולטיבי לפני תחילת העונה
                
                # עזר למציאת קבוצה ב-X_all ובטבלה
                def get_standings_team(raw_name):
                    norm = normalize_team_name(raw_name)
                    # Use our robust mapping to try and normalize
                    if norm in TEAM_MAPPING_TO_CSV:
                        mapped_csv = TEAM_MAPPING_TO_CSV[norm]
                        # map back to standings if needed? Wait, standings has points_sim.
                        # It's actually better to just check against points_sim keys
                        
                    keys = list(points_sim.keys())
                    if norm in keys:
                        return norm
                    
                    # try difflib
                    m = difflib.get_close_matches(norm, keys, n=1, cutoff=0.4)
                    if m:
                        return m[0]
                    
                    st.error(f"Missing team in standings: {raw_name} -> {norm}")
                    return norm

                for f in fixtures:
                    home_norm = get_standings_team(f['home_team_norm'])
                    away_norm = get_standings_team(f['away_team_norm'])
                    
                    # כוח כושר באקספוננציאל (Power Law)
                    ppg_home = max(ppg_dict.get(home_norm, 1.0), 0.1)
                    ppg_away = max(ppg_dict.get(away_norm, 1.0), 0.1)
                    home_power = ppg_home ** 2.5
                    away_power = ppg_away ** 2.5
                    home_form_share = home_power / (home_power + away_power)
                    away_form_share = away_power / (home_power + away_power)
                    
                    csv_home = get_csv_team_name(home_norm, clubs)
                    csv_away = get_csv_team_name(away_norm, clubs)
                    
                    # מנגנון הצלה למודל (Missing Data Fallback) במקום לזרוק Exception נאפל לערך ממוצע
                    if csv_home and csv_home in X_all.index:
                        x_h = X_all.loc[[csv_home]]
                    else:
                        x_h = X_all.mean(axis=0).to_frame().T
                        
                    if csv_away and csv_away in X_all.index:
                        x_a = X_all.loc[[csv_away]]
                    else:
                        x_a = X_all.mean(axis=0).to_frame().T
                    
                    p_home = float(model.predict_proba(x_h)[0][1])
                    p_away = float(model.predict_proba(x_a)[0][1])
                    
                    # חישוב הסתברויות גולמיות (מודל היסטורי)
                    hw_prob_raw, d_prob_raw, aw_prob_raw = compute_match_outcome_probs(p_home, p_away, home_advantage=0.0)

                    
                    # שקלול מתוקן (35% מודל, 65% כושר נוכחי)
                    final_home_prob = (0.35 * hw_prob_raw) + (0.65 * home_form_share)
                    final_away_prob = (0.35 * aw_prob_raw) + (0.65 * away_form_share)
                    final_draw_prob = d_prob_raw * 0.85
                    
                    # נרמול ל-1.0
                    total_prob = final_home_prob + final_draw_prob + final_away_prob
                    hw_final = final_home_prob / total_prob
                    d_final = final_draw_prob / total_prob
                    aw_final = final_away_prob / total_prob
                    
                    # חלוקת נקודות לפי תוחלת (Expected Value)
                    home_expected_points = (hw_final * 3) + (d_final * 1)
                    away_expected_points = (aw_final * 3) + (d_final * 1)
                    
                    points_sim[home_norm] = points_sim.get(home_norm, 0) + home_expected_points
                    points_sim[away_norm] = points_sim.get(away_norm, 0) + away_expected_points
                            
                    simulated_games[home_norm] = simulated_games.get(home_norm, 0) + 1
                    simulated_games[away_norm] = simulated_games.get(away_norm, 0) + 1
                        
                # בדיקה והשלמה ל-38 משחקים (נגד יריבה ממוצעת וירטואלית)
                average_team_features = X_all.mean(axis=0).to_frame().T
                avg_p = float(model.predict_proba(average_team_features)[0][1])
                
                for norm, played in played_games.items():
                    total_simulated = simulated_games.get(norm, 0)
                    total_games = played + total_simulated
                    
                    if total_games < 38:
                        missing = 38 - total_games
                        csv_team = get_csv_team_name(norm, clubs)
                        
                        # כוח כושר באקספוננציאל (Power Law) לווירטואלית
                        ppg_team = max(ppg_dict.get(norm, 1.0), 0.1)
                        ppg_virtual = 1.0
                        team_power = ppg_team ** 2.5
                        virtual_power = ppg_virtual ** 2.5
                        team_form_share = team_power / (team_power + virtual_power)
                        virtual_form_share = virtual_power / (team_power + virtual_power)
                        
                        if csv_team and csv_team in X_all.index:
                            x_team = X_all.loc[[csv_team]]
                        else:
                            x_team = average_team_features
                            
                        p_team = float(model.predict_proba(x_team)[0][1])
                        hw_prob_raw, d_prob_raw, aw_prob_raw = compute_match_outcome_probs(p_team, avg_p, home_advantage=0.0)
                        
                        for _ in range(missing):
                            # שקלול מתוקן (35% מודל, 65% כושר נוכחי)
                            final_team_prob = (0.35 * hw_prob_raw) + (0.65 * team_form_share)
                            final_virtual_prob = (0.35 * aw_prob_raw) + (0.65 * virtual_form_share)
                            final_draw_prob = d_prob_raw * 0.85
                            
                            # נרמול ל-1.0
                            total_prob = final_team_prob + final_draw_prob + final_virtual_prob
                            hw_final = final_team_prob / total_prob
                            d_final = final_draw_prob / total_prob
                            aw_final = final_virtual_prob / total_prob
                            
                            # חלוקת נקודות לפי תוחלת (Expected Value)
                            team_expected_points = (hw_final * 3) + (d_final * 1)
                            points_sim[norm] += team_expected_points
                            
                            simulated_games[norm] += 1

                # בניית הטבלה הסופית (ללא עמודת בקרה)
                final_rows = []
                for _, row in live_standings_df.iterrows():
                    norm = row['team_name_norm']
                    final_rows.append({
                        "קבוצה": row['team_name'],
                        "נקודות סופיות": round(points_sim.get(norm, row['points']))
                    })
                
                final_df = pd.DataFrame(final_rows)
                final_df = final_df.sort_values(by="נקודות סופיות", ascending=False).reset_index(drop=True)
                final_df.index = final_df.index + 1 # 1-indexed
                
                # צביעת שורות Pandas Styler
                def highlight_table(s):
                    if s.name == 1:
                        return ['background-color: rgba(255, 215, 0, 0.3); color: white;'] * len(s) # Gold
                    elif 2 <= s.name <= 4:
                        return ['background-color: rgba(76, 175, 80, 0.3); color: white;'] * len(s) # CL green
                    elif 18 <= s.name <= 20:
                        return ['background-color: rgba(244, 67, 54, 0.3); color: white;'] * len(s) # Relegation red
                    return [''] * len(s)

                styled_df = final_df.style.apply(highlight_table, axis=1)
                
                st.dataframe(styled_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"שגיאה במהלך הסימולציה: {e}")
                
st.markdown('</div>', unsafe_allow_html=True)
