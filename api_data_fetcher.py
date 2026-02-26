"""
football-data.org data fetcher
==============================

This module provides helper functions to fetch **live Premier League data**
from the free football-data.org API (https://www.football-data.org/)
and convert the JSON responses into **Pandas DataFrames**.

Security / API key handling (חשוב מאוד):
----------------------------------------
- NEVER hard-code your API key in the source code.
- We use the `python-dotenv` package to load the key from a local `.env` file.

How to set up the .env file (בצע פעם אחת מקומית):
1. Install dependencies:

   pip install requests python-dotenv pandas

2. In the project root (next to this file), create a file named `.env`
   with the following content (replace YOUR_KEY_HERE with your real key):

   FOOTBALL_DATA_ORG_KEY=YOUR_KEY_HERE

3. Make sure `.env` is **not** committed to git:
   - Add the following line to your `.gitignore` (in the repo root):

     .env

   This ensures your secret API key will never be pushed to GitHub.

Later, the Streamlit app (or any other module) can import and call
`fetch_premier_league_standings_df` to get a fresh DataFrame with
live league table data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import os
import difflib
import re
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv


# ==========================
# Configuration / constants
# ==========================

API_BASE_URL = "https://api.football-data.org/v4"


@dataclass
class APIFootballConfig:
    """
    Simple configuration holder for API-Football credentials.
    """

    api_key: str


def load_api_config(env_var_name: str = "FOOTBALL_DATA_ORG_KEY") -> APIFootballConfig:
    """
    Loads the API key from the .env file using python-dotenv.

    Steps:
    1. `load_dotenv()` reads environment variables from the local `.env` file.
    2. We read the value of `env_var_name` (default: "API_FOOTBALL_KEY").
    3. If it's missing, we raise a clear error so the user knows what to fix.

    This function can be reused by Streamlit or other scripts to ensure
    the API key is always loaded in a consistent and secure way.
    """

    # טוען את משתני הסביבה מתוך קובץ .env (אם קיים)
    load_dotenv()

    api_key = os.getenv(env_var_name)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{env_var_name}' is not set.\n"
            "Create a .env file in your project root with a line like:\n"
            "FOOTBALL_DATA_ORG_KEY=YOUR_KEY_HERE\n"
            "and make sure '.env' is listed in your .gitignore."
        )

    return APIFootballConfig(api_key=api_key)


def _api_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[APIFootballConfig] = None,
) -> Dict[str, Any]:
    """
    Internal helper for sending GET requests to football-data.org.

    - `endpoint`: path after the base URL, e.g. "/competitions/PL/standings".
    - `params`: query parameters for the request.
    - `config`: optional APIFootballConfig. If None, it will be loaded
      from the environment via `load_api_config`.

    Returns the parsed JSON as a Python dict and raises an exception
    for non-200 HTTP status codes.
    """

    if config is None:
        config = load_api_config()

    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    headers = {
        # football-data.org uses X-Auth-Token header for authentication
        "X-Auth-Token": config.api_key,
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()
    if "errors" in data and data["errors"]:
        raise RuntimeError(f"API returned errors: {data['errors']}")

    return data


def _extract_standings_to_df(api_response: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the JSON response from football-data.org into a tidy Pandas DataFrame.

    The typical structure for /competitions/PL/standings looks like:

    {
      "standings": [
        {
          "stage": "REGULAR_SEASON",
          "type": "TOTAL",
          "group": null,
          "table": [
            {
              "position": 1,
              "team": { "id": ..., "name": "Arsenal FC", ... },
              "playedGames": 38,
              "won": 28,
              "draw": 2,
              "lost": 8,
              "goalsFor": 86,
              "goalsAgainst": 40,
              "goalDifference": 46,
              "points": 86,
              "form": "W,W,D,L,W",
              ...
            },
            ...
          ]
        }
      ]
    }

    IMPORTANT: the returned DataFrame MUST contain at least the following
    columns, which are used by the Streamlit app:
    - rank
    - team_name
    - played
    - points
    - form
    """

    standings_list: List[Any] = api_response.get("standings", [])
    if not standings_list:
        raise ValueError("API response has no 'standings' data.")

    first_standings_block = standings_list[0]
    table_entries: List[Dict[str, Any]] = first_standings_block.get("table", [])
    if not table_entries:
        raise ValueError("API response has no 'table' data inside 'standings'.")

    rows: List[Dict[str, Any]] = []
    for team_entry in table_entries:
        team_info = team_entry.get("team", {}) or {}

        rows.append(
            {
                # required by the rest of the project
                "rank": team_entry.get("position"),
                "team_id": team_info.get("id"),
                "team_name": team_info.get("name"),
                "team_logo": None,  # football-data.org free tier does not expose logos here
                "played": team_entry.get("playedGames"),
                "wins": team_entry.get("won"),
                "draws": team_entry.get("draw"),
                "losses": team_entry.get("lost"),
                "goals_for": team_entry.get("goalsFor"),
                "goals_against": team_entry.get("goalsAgainst"),
                "goal_diff": team_entry.get("goalDifference"),
                "points": team_entry.get("points"),
                "group": first_standings_block.get("group"),
                "form": team_entry.get("form"),
                "status": team_entry.get("status"),
                "description": first_standings_block.get("stage"),
                "update": None,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ==========================
# Web scraping for Form (Last 5 games)
# ==========================


def normalize_team_name(name: str) -> str:
    """
    מנרמל שם קבוצה כדי לצמצם בעיות התאמה בין שמות מה-API לשמות מה-scraping.

    פעולות:
    - המרה לאותיות קטנות.
    - החלפות קבועות לכינויים מוכרים (Aliasing).
    - החלפת מקפים/קו מפריד ברווח.
    - הסרת עוגנים קבועים כמו fc, afc.
    - הסרת נקודות ורווחים כפולים.
    """

    if not isinstance(name, str):
        return ""

    s = name.lower()
    
    # תרגום חריגים ידועים מה-API לקבוצות שלנו
    aliases = {
        "man united": "manchester united",
        "man utd": "manchester united",
        "manchester united fc": "manchester united",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "nott'm forest": "nottingham forest",
        "nottingham": "nottingham forest",
        "wolves": "wolverhampton wanderers",
        "wolverhampton": "wolverhampton wanderers",
        "newcastle": "newcastle united",
        "newcastle utd": "newcastle united",
        "brighton": "brighton hove albion",
        "brighton & hove albion fc": "brighton hove albion",
        "leicester": "leicester city",
        "leeds": "leeds united",
        "west ham": "west ham united",
        "aston villa": "aston villa",
        "aston villa fc": "aston villa",
        "arsenal fc": "arsenal",
        "chelsea fc": "chelsea",
        "liverpool fc": "liverpool",
        "manchester city fc": "manchester city"
    }
    
    for k, v in aliases.items():
        if s == k:
            s = v
            break # No need to continue if exact match alias found
            
    for ch in ["-", "–", "_", "&"]:
        s = s.replace(ch, " ")
    for ch in ["."]:
        s = s.replace(ch, "")
        
    s = s.replace(" fc", "").replace(" afc", "").replace(" utd", " united")
    
    s = " ".join(s.split())
    return s


def _scrape_premier_league_form() -> pd.DataFrame:
    """
    מגרד (scrape) את טבלת הפרמייר ליג מאתר ציבורי (כאן: Sky Sports)
    ומחזיר DataFrame עם עמודות: team_name_scrape, form_scrape.

    הערות:
    - משתמשים ב-requests עם User-Agent מזויף כדי לצמצם סיכוי ל-403.
    - משתמשים ב-pandas.read_html כדי לחלץ את הטבלה המתאימה מה-HTML.
    """

    url = "https://www.bbc.com/sport/football/tables"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"FBref scraping failed with status code {resp.status_code}"
        )

    # קוראים את כל הטבלאות מהעמוד ולוקחים את הראשונה (BBC מחזיקה את טבלת הליגה שם)
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("Could not find any tables on BBC football tables page.")

    target_df = tables[0]

    # זיהוי שמות העמודות הרלוונטיים:
    # - עמודת קבוצה: מכילה 'team' או 'squad' בשם, ואם לא – ניקח את העמודה השלישית (אינדקס 2).
    # - עמודת Form: מכילה 'form' בשם, ואם לא – ניקח את העמודה האחרונה.
    cols_lower = [str(c).lower() for c in target_df.columns]
    team_col = None
    form_col = None

    for original, lower in zip(target_df.columns, cols_lower):
        if ("team" in lower) or ("squad" in lower):
            team_col = original
        if "form" in lower:
            form_col = original

    if team_col is None:
        # ברירת מחדל: העמודה השלישית
        try:
            team_col = target_df.columns[2]
        except IndexError:
            raise RuntimeError("Could not identify a team column in BBC table.")

    if form_col is None:
        # ברירת מחדל: העמודה האחרונה
        form_col = target_df.columns[-1]

    df_form = target_df[[team_col, form_col]].copy()
    df_form.columns = ["team_name_scrape", "form_scrape"]
    df_form = df_form.dropna(subset=["team_name_scrape"])
    df_form["team_name_norm"] = df_form["team_name_scrape"].apply(normalize_team_name)

    # ניקוי המחרוזת כך שיישארו רק האותיות הגדולות W, D, L, ומתוכן 5 האחרונות
    df_form["form_scrape"] = df_form["form_scrape"].astype(str).apply(
        lambda x: re.sub(r'[^WDL]', '', x)[-5:]
    )

    print(f"[DEBUG] Scraping form data succeeded. Scraped {len(df_form)} teams from BBC Sport.")
    return df_form


def _merge_form_from_scraper(df: pd.DataFrame) -> pd.DataFrame:
    """
    ממזג את נתוני ה-Form (רצף המשחקים) שנגרדו מ-FBref לתוך DataFrame
    שהגיע מה-API של football-data.org.

    - משתמש ב-normalize_team_name + difflib.get_close_matches כדי להתאים שמות.
    - אם ה-scraping נכשל, מחזיר את df המקורי ללא שינוי.
    """

    try:
        df_form = _scrape_premier_league_form()
    except Exception as e:
        print(f"[DEBUG] Scraping form data failed: {e}")
        return df

    df = df.copy()
    df["team_name_norm"] = df["team_name"].apply(normalize_team_name)

    # בניית מיפוי משם מנורמל ל-form_scrape
    form_map: Dict[str, Any] = {}
    main_norm_names = df["team_name_norm"].tolist()

    for _, row in df_form.iterrows():
        src_norm = row["team_name_norm"]
        form_val = row["form_scrape"]
        if not src_norm:
            continue
        matches = difflib.get_close_matches(src_norm, main_norm_names, n=1, cutoff=0.6)
        if matches:
            target_norm = matches[0]
            form_map[target_norm] = form_val

    df["form"] = df["team_name_norm"].map(form_map)

    matched_count = df["form"].notna().sum()
    print(
        f"[DEBUG] Merged scraped form data into API standings. "
        f"Matched {matched_count} teams."
    )

    return df


def fetch_premier_league_standings_df(
    season: Optional[int] = None,
    config: Optional[APIFootballConfig] = None,
) -> pd.DataFrame:
    """
    Fetches the current Premier League standings (competition "PL") from
    football-data.org and returns them as a Pandas DataFrame.

    Parameters
    ----------
    season : int, optional
        Kept for backwards compatibility but currently unused for this endpoint.
    config : APIFootballConfig, optional
        Optionally pass a pre-loaded config. If None, it will be loaded
        from the environment.

    This function is designed to be imported directly into the Streamlit app,
    so the UI can display **live league table data** instead of a static CSV.
    """

    # football-data.org exposes the PL standings directly via this endpoint.
    raw = _api_get("/competitions/PL/standings", params=None, config=config)
    df = _extract_standings_to_df(raw)
    # ניסיון להעשיר את טבלת הליגה בנתוני Form חיים באמצעות Web Scraping
    df = _merge_form_from_scraper(df)
    return df


def fetch_remaining_fixtures(config: Optional[APIFootballConfig] = None) -> List[Dict[str, str]]:
    """
    מביא את כל המשחקים שעוד לא שוחקו בפרמייר ליג.
    מחזיר רשימה של מילונים עם שמות הקבוצות מנורמלים.
    """
    raw = _api_get("/competitions/PL/matches", params=None, config=config)
    matches = raw.get("matches", [])
    
    fixtures = []
    for match in matches:
        if match.get("status") == "FINISHED":
            continue
            
        home_team = match.get("homeTeam", {}).get("name")
        away_team = match.get("awayTeam", {}).get("name")
        if home_team and away_team:
            fixtures.append({
                "home_team_norm": normalize_team_name(home_team),
                "away_team_norm": normalize_team_name(away_team),
                "home_team_raw": home_team,
                "away_team_raw": away_team
            })
    return fixtures


if __name__ == "__main__":
    """
    דוגמה להרצה עצמאית מהטרמינל:

    python api_data_fetcher.py

    זה ידפיס את טבלת הליגה המעודכנת של הפרמייר ליג (אם ההרשאות וה-API Key תקינים).
    """

    cfg = load_api_config()
    table_df = fetch_premier_league_standings_df(config=cfg)
    print(table_df)

