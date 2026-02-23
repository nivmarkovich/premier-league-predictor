"""
סקריפט דוגמה: מודל פשוט לחיזוי "חוזק קבוצה" בפרמייר ליג
בהשראת המחברת מ-Kaggle (english-premier-league-players-statistics.ipynb).

הרעיון:
- להשתמש בסטטיסטיקות שחקנים כדי לבנות פיצ'רים ברמת קבוצה (Club).
- להגדיר "קבוצה חזקה" כקבוצה עם אחוז ניצחונות מעל החציון בליגה.
- לאמן מודל סיווג (Logistic Regression) שמנסה לנבא אם קבוצה היא "חזקה" או "פחות חזקה".
- זה לא חיזוי ישיר של תוצאת משחק בודד, אלא צעד ראשון הגיוני:
  ברגע שיש ציון "חוזק" לכל קבוצה, אפשר להשוות בין שתי קבוצות כדי להעריך מי פייבוריט.

איך להריץ את הסקריפט:
1. להוריד את קובץ ה-CSV מהקגל (אותו קובץ שהמחברת משתמשת בו)
   ולשמור אותו, למשל, תחת:
   data/premier_league_players.csv
2. לפתוח טרמינל/PowerShell בתוך תיקיית הפרויקט הזו.
3. להריץ:
   python premier_league_team_strength_model.py
   (או: python path/to/file.py אם השם/המיקום שונים)

אפשר גם להעביר נתיב לקובץ כארגומנט:
   python premier_league_team_strength_model.py "c:\\path\\to\\dataset.csv"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==========================
# הגדרות כלליות
# ==========================

# ברירת מחדל לנתיב קובץ ה-CSV (תיקיית data מקומית)
DEFAULT_CSV_PATH = Path("data") / "premier_league_players.csv"


def load_player_data(csv_path: Path) -> pd.DataFrame:
    """
    טוען את קובץ השחקנים מתוך נתיב שניתן.

    למה פונקציה נפרדת?
    - כדי להפריד בין "I/O" לבין לוגיקת העיבוד.
    - זה הופך את הקוד לנקי יותר וקל לבדיקה.

    אם הקובץ לא קיים – נזרקת שגיאה ברורה, כדי שתדע לשנות את הנתיב.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"לא נמצא קובץ נתונים בנתיב: {csv_path}.\n"
            f"בדוק שהורדת את קובץ ה-CSV מה-Kaggle ושמרת אותו בנתיב הזה."
        )

    df = pd.read_csv(csv_path)
    return df.copy()  # עותק כדי לא לדרוס את המקור בטעות


def preprocess_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    עושה את עיבוד המידע הבסיסי כמו במחברת:

    1. מסיר רשומות בלי Nationality / Age / Jersey Number – אלו רשומות בעייתיות.
    2. מנקה עמודות אחוזים (מחרוזות כמו '45%') והופך אותן לערכים מספריים (float).
    3. יוצר גרסה "פר משחק" (per-game) של הרבה סטטיסטיקות על ידי חלוקה במספר הופעות.
       זה קריטי כדי להשוות שחקנים ששיחקו מספר שונה של משחקים.
    4. מסנן רק שחקנים עם לפחות 38 הופעות (עונה מלאה) כדי להפחית רעש.

    איך זה עוזר נגד Overfitting?
    - נרמול per-game מפחית קיצוניות מלאכותית של שחקנים עם מעט דקות משחק.
    - סינון לשחקנים עם מספיק משחקים מוריד "רעש" ו-outliers -> מודל לומד דפוסים יציבים יותר.
    """

    # 1. הסרת רשומות עם ערכים חסרים קריטיים
    df = df[df["Nationality"].notna()]
    df = df[df["Age"].notna()]
    df = df[df["Jersey Number"].notna()]

    # 2. ניקוי אחוזים והמרה ל-float
    percent_cols = [
        "Cross accuracy %",
        "Shooting accuracy %",
        "Tackle success %",
    ]

    for col in percent_cols:
        if col in df.columns:
            # מחליף את סימן האחוזים + המרה ל-float
            df[col] = (
                df[col]
                .astype(str)         # להבטיח שהערכים הם מחרוזת
                .str.replace("%", "")  # הסרת סימן %
                .replace("nan", np.nan)
                .astype(float)
            )

    # 3. בניית DataFrame "נקי" עם כל העמודות
    features = df.columns
    data_clean = df[features].copy()

    # הסרת שחקנים ללא הופעות (כדי לא לחלק ב-0)
    data_clean_app_non_zero = data_clean[data_clean["Appearances"] > 0]

    # עמודות שלא נחלק ב-"Appearances" (זה כמו במחברת, אבל בקירוב)
    cols_not_divided = [
        "Age",
        "Name",
        "Appearances",
        "Club",
        "Nationality",
        "Jersey Number",
        "Cross accuracy %",
        "Position",
        "Goals per match",
        "Passes per match",
        "Tackle success %",
        "Shooting accuracy %",
    ]
    cols_to_divide = [
        c for c in features if c not in cols_not_divided and c in data_clean_app_non_zero.columns
    ]

    # חלוקה per-game במכה אחת (המרה ל-float כדי למנוע שגיאות Pandas)
    data_clean_app_non_zero[cols_to_divide] = (
        data_clean_app_non_zero[cols_to_divide]
        .astype(float)
        .div(data_clean_app_non_zero["Appearances"], axis=0)
    )

    # 4. שמירה רק על שחקנים עם לפחות 38 הופעות (יציבות סטטיסטית)
    data_clean_app_non_zero = data_clean_app_non_zero[
        data_clean_app_non_zero["Appearances"] >= 38
    ]

    return data_clean_app_non_zero


def build_team_level_features(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    אגרגציה מרמת שחקן לרמת קבוצה (Club).

    שלבים:
    - בחירת עמודות מספריות בלבד (פיצ'רים).
    - groupby לפי Club ולקיחת ממוצע per-game של כל הפיצ'רים המספריים.
    - חישוב win_rate (ניצחונות / הופעות) ברמת קבוצה על בסיס נתוני השחקנים.
      זה ישמש לנו כ-target לצורך המודל.

    למה זה חשוב ל-Overfitting?
    - מעבר מרמת שחקן לרמת קבוצה מדלל מאוד את כמות הדגימות,
      אבל כל דגימה "עמוקה" ועשירה בפיצ'רים ממוצעים -> פחות רעש.
    - המודל לא מנסה לזכור כל שחקן בנפרד אלא את דפוסי הקבוצה כיחידה.
    """

    # עמודות מספריות בלבד
    numeric_df = df_players.select_dtypes(include=[np.number]).copy()

    # הוספת העמודה Club לפיצ'רים המספריים
    numeric_df["Club"] = df_players["Club"].values

    # אגרגציה: ממוצע per-game לכל קבוצה
    team_features = numeric_df.groupby("Club").mean()

    # חישוב מדדי ניצחונות ברמת קבוצה (מסכמים את Wins ו-Appearances לכל שחקני הקבוצה)
    wins_by_club = df_players.groupby("Club")["Wins"].sum()
    apps_by_club = df_players.groupby("Club")["Appearances"].sum()

    win_rate = wins_by_club / apps_by_club
    team_features["win_rate"] = win_rate

    # הסרת קבוצות עם ערכי win_rate חסרים/אינסופיים
    team_features = team_features.replace([np.inf, -np.inf], np.nan).dropna(subset=["win_rate"])

    return team_features


def build_train_test(team_features: pd.DataFrame):
    """
    בונה X (פיצ'רים) ו-y (תגית) ומחלק ל-train/test.

    עיצוב הבעיה:
    - נגדיר "קבוצה חזקה" (label = 1) אם אחוז הניצחונות שלה (win_rate)
      גדול או שווה לחציון (median) הליגה.
    - השאר יקבלו 0 (קבוצה פחות חזקה).

    איך זה מונע Overfitting?
    - חלוקה ל-train/test (עם random_state קבוע) מבטיחה שנמדוד ביצועים על נתונים
      שהמודל לא ראה באימון.
    - שימוש ביעד פשוט (high/low) במקום רגרסיה על מספר מדויק מונע ניסיון
      של המודל "לרדוף אחרי רעש" קטן.
    """

    # הגדרת ה-label
    median_win_rate = team_features["win_rate"].median()
    y = (team_features["win_rate"] >= median_win_rate).astype(int)

    # X – כל הפיצ'רים המספריים חוץ מה-label
    feature_cols = [c for c in team_features.columns if c != "win_rate"]
    X = team_features[feature_cols]

    # טיפול ב-NAs שנשארו (פשוט: למלא ב-0 כדי לשמור את הדגימות)
    X = X.fillna(0.0)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y,  # שומר על יחס קלאסים דומה ב-train ו-test
    )

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train) -> Pipeline:
    """
    מאמן Pipeline של:
    StandardScaler -> LogisticRegression

    למה בחירה כזו טובה לפרויקט CV וכיצד היא מונעת Overfitting?

    - StandardScaler:
      מנרמל את כל הפיצ'רים לאותה סקלה (ממוצע 0, סטיית תקן 1),
      כך שאף פיצ'ר בודד לא "ישלוט" על הפונקציה המפרידה.
      זה עוזר למודל להתכנס ולבנות משקולות מאוזנות.
    - LogisticRegression עם penalty='l2':
      זה מודל ליניארי עם רגולריזציה L2 (ברירת מחדל ב-sklearn),
      כלומר הוא 'מעונש' על משקולות גדולות מדי.
      התוצאה: פחות Overfitting, יותר הכללה לנתונים חדשים.
    - המודל יחסית פשוט ומוסבר (explainable), מה שנראה מקצועי ב-CV
      אבל עדיין קריא וקל להבנה.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,        # עוצמת רגולריזציה (ערך נמוך יותר = יותר רגולריזציה)
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test, y_test):
    """
    מדפיס מדדי ביצוע בסיסיים של המודל על סט הבדיקה (test set).

    זה שלב קריטי נגד Overfitting:
    - אנחנו מתעניינים בדיוק בביצועים על נתונים שהמודל *לא* ראה באימון.
    - אם הדיוק ב-train מאוד גבוה וב-test נמוך, סימן שהמודל "שינן" את הנתונים (Overfit).
    """

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("ביצועי המודל על סט הבדיקה (קבוצות שלא נראו באימון):")
    print(f"Accuracy: {acc:.3f}")
    print()
    print("דו\"ח סיווג מפורט:")
    print(classification_report(y_test, y_pred, target_names=["קבוצה פחות חזקה", "קבוצה חזקה"]))
    print("=" * 60)


def main():
    """
    הפונקציה הראשית שמריצה את כל ה-Pipeline:

    1. קריאת קובץ ה-CSV (נתיב ברירת מחדל או נתיב מה-CLI).
    2. עיבוד וניקוי הנתונים ברמת שחקן (per-game, סינון הופעות).
    3. מעבר לרמת קבוצה ואגרגציה של פיצ'רים.
    4. בניית סט אימון ובדיקה.
    5. אימון מודל Logistic Regression עם רגולריזציה.
    6. הערכת ביצועים על סט הבדיקה.
    7. הדפסה קצרה של "איך להשתמש בזה לרעיון של חיזוי משחקים".

    זה מאפשר לך להראות ב-CV:
    - קוד מסודר, מודולרי ונקי.
    - התייחסות מפורשת ל-Overfitting והפרדת train/test.
    - שימוש בספריות סטנדרטיות (pandas, sklearn).
    """

    # ==========================
    # 1. קריאת נתיב קובץ מה-CLI (אם סופק)
    # ==========================
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[0]).parent / sys.argv[1]
    else:
        csv_path = DEFAULT_CSV_PATH

    print(f"קורא נתונים מ: {csv_path}")
    df_players_raw = load_player_data(csv_path)

    # ==========================
    # 2. עיבוד וניקוי ברמת שחקן
    # ==========================
    df_players_clean = preprocess_players(df_players_raw)
    print(f"מספר שחקנים אחרי ניקוי וסינון (Appearances >= 38): {len(df_players_clean)}")

    # ==========================
    # 3. בניית פיצ'רים ברמת קבוצה
    # ==========================
    team_features = build_team_level_features(df_players_clean)
    print(f"מספר קבוצות במודל: {len(team_features)}")

    # ==========================
    # 4. בניית train/test
    # ==========================
    X_train, X_test, y_train, y_test, feature_cols = build_train_test(team_features)

    # ==========================
    # 5. אימון מודל
    # ==========================
    model = train_model(X_train, y_train)

    # ==========================
    # 6. הערכת ביצועים
    # ==========================
    evaluate_model(model, X_test, y_test)

    # ==========================
    # 7. הדגמה רעיונית: שימוש בחיזוי "חוזק קבוצה"
    # ==========================
    print()
    print("איך מחברים את זה לרעיון של חיזוי תוצאות משחקים?")
    print("- המודל נותן לכל קבוצה הסתברות להיות 'חזקה'.")
    print("- כדי לחזות משחק בין שתי קבוצות, אפשר:")
    print("  1. לקחת את הפיצ'רים של כל קבוצה (מאותו team_features).")
    print("  2. להריץ model.predict_proba(X_single_team) לכל אחת.")
    print("  3. הקבוצה עם הסתברות גבוהה יותר להיות 'חזקה' היא הפייבוריט התיאורטי.")
    print("- לפרויקט עמוק יותר אפשר לצרף דאטה סט של תוצאות משחקים אמיתיות,")
    print("  לבנות שורות ברמת משחק (בית/חוץ, פער חוזק, נתוני כושר נוכחי וכו'),")
    print("  ולהחליף את ה-label ל: Win/Draw/Loss ולבנות מודל ישירות ברמת משחק.")


if __name__ == "__main__":
    main()