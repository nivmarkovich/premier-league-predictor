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

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# מייבאים פונקציות ולוגיקה מהסקריפט הקיים
from premier_league_team_strength_model import (
    DEFAULT_CSV_PATH,
    load_player_data,
    preprocess_players,
    build_team_level_features,
    train_model,
)


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

    return X, y, clubs, feature_cols, team_features


@st.cache_resource
def train_cached_model(csv_path_str: str):
    """
    מאמן את המודל פעם אחת ושומר אותו ב-cache של Streamlit.

    למה cache_resource?
    - אימון מודל לוקח זמן יחסי, ואין צורך לאמן מחדש בכל רענון UI.
    - ברגע שהמודל אומן פעם אחת לנתיב נתונים מסוים, אפשר להשתמש בו
      שוב ושוב בקריאות predict בלי המתנה מיותרת.
    """

    X, y, clubs, feature_cols, team_features = prepare_team_data(csv_path_str)

    # חלוקה ל-train/test כדי לשמור את אותם עקרונות נגד Overfitting
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    # חישוב דיוק בסיסי על סט הבדיקה לצורך הצגה למשתמש
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, X, y, clubs, feature_cols, team_features, acc


# ==========================
# הגדרת ממשק המשתמש ב-Streamlit
# ==========================

st.set_page_config(
    page_title="Premier League Team Strength Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ חיזוי חוזק קבוצות בפרמייר ליג")
st.write(
    "אפליקציה קטנה שמבוססת על נתוני השחקנים וסטטיסטיקות היסטוריות.\n"
    "המודל מנבא לכל קבוצה הסתברות להיות מוגדרת כ*קבוצה חזקה*, "
    "ועל בסיס זה מעריך מי הפייבוריט התיאורטי במשחק ביניהן."
)

# בחירת נתיב לקובץ ה-CSV (עם ברירת מחדל מהסקריפט המקורי)
st.sidebar.header("הגדרות נתונים")
default_path_str = str(DEFAULT_CSV_PATH)
csv_path_input = st.sidebar.text_input(
    "נתיב לקובץ השחקנים (CSV)",
    value=default_path_str,
    help="ברירת מחדל: אותו נתיב שמוגדר בקובץ premier_league_team_strength_model.py",
)

try:
    # מאמנים את המודל (או טוענים מה-cache)
    model, X_all, y_all, clubs, feature_cols, team_features, test_accuracy = train_cached_model(
        csv_path_input
    )
except FileNotFoundError as e:
    st.error(
        "לא הצלחתי למצוא את קובץ הנתונים.\n\n"
        f"{e}\n\n"
        "עדכן את הנתיב בצד שמאל לקובץ CSV הנכון (כפי שהורדת מ-Kaggle)."
    )
    st.stop()
except Exception as e:
    st.error(f"קרתה שגיאה בזמן טעינת הנתונים או אימון המודל: {e}")
    st.stop()

# הצגת מידע כללי על המודל
st.sidebar.subheader("מידע על המודל")
st.sidebar.write(f"דיוק על סט הבדיקה (test accuracy): **{test_accuracy:.2%}**")
st.sidebar.write(f"מספר קבוצות בדאטה: **{len(clubs)}**")

st.markdown("---")
st.subheader("בחר שתי קבוצות להשוואה")

if len(clubs) < 2:
    st.warning("נדרשות לפחות שתי קבוצות בדאטה כדי לבצע השוואה.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("קבוצה 1", clubs, index=0)

with col2:
    # בחירת קבוצה 2, ברירת מחדל – הקבוצה השנייה ברשימה אם קיימת
    default_index = 1 if len(clubs) > 1 else 0
    team_b = st.selectbox("קבוצה 2", clubs, index=default_index)

st.markdown(
    "_הערה_: המודל לא רואה מי בית ומי חוץ, "
    "רק חוזק כללי של כל קבוצה על בסיס היסטוריית המשחקים."
)

if team_a == team_b:
    st.warning("בחר שתי קבוצות שונות כדי לבצע השוואה.")
    st.stop()

if st.button("חשב הסתברות לכל קבוצה"):
    # מוציאים את השורות המתאימות לכל קבוצה מ-X_all
    try:
        idx_a = clubs.index(team_a)
        idx_b = clubs.index(team_b)
    except ValueError:
        st.error("לא הצלחתי למצוא את אחת הקבוצות ברשימת הפיצ'רים.")
        st.stop()

    X_team_a = X_all.iloc[[idx_a]]
    X_team_b = X_all.iloc[[idx_b]]

    # predict_proba מחזיר הסתברות לכל מחלקה; מחלקה 1 היא "חזקה"
    proba_a = float(model.predict_proba(X_team_a)[0][1])
    proba_b = float(model.predict_proba(X_team_b)[0][1])

    st.markdown("### תוצאות החיזוי")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(
            label=f"הסתברות להיות 'קבוצה חזקה' – {team_a}",
            value=f"{proba_a:.2%}",
        )
    with col_res2:
        st.metric(
            label=f"הסתברות להיות 'קבוצה חזקה' – {team_b}",
            value=f"{proba_b:.2%}",
        )

    st.markdown("---")

    # קביעה מי "פייבוריט" לפי הסתברות גבוהה יותר
    eps = 1e-3  # טולרנס קטן בשביל הבדלים זניחים
    if abs(proba_a - proba_b) < eps:
        st.info(
            "לפי המודל, שתי הקבוצות כמעט שוות בחוזק שלהן – "
            "קשה להגיד מי פייבוריט מובהק."
        )
    elif proba_a > proba_b:
        st.success(
            f"לפי המודל, **{team_a}** היא הקבוצה היותר חזקה ולכן הפייבוריט התיאורטי במשחק הזה."
        )
    else:
        st.success(
            f"לפי המודל, **{team_b}** היא הקבוצה היותר חזקה ולכן הפייבוריט התיאורטי במשחק הזה."
        )

    st.markdown(
        "זו כמובן הערכה גסה בלבד, המבוססת על סטטיסטיקות היסטוריות של השחקנים בלבד "
        "ולא על פקטורים כמו פציעות, בית/חוץ, כושר נוכחי ועוד."
    )

