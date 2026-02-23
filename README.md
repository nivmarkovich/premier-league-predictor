## Premier League Team Strength Predictor

A small end-to-end ML project that predicts the **relative strength of Premier League teams** based on **player-level statistics**, with an interactive **Streamlit** web app for exploring matchups between any two clubs.

### Overview

This project takes historical Premier League player statistics from Kaggle and:

- **Aggregates player stats into team-level features** (per club).
- **Trains a logistic regression model** to classify whether a team is “strong” or “not strong” based on its historical performance.
- Exposes a **Streamlit UI** where you can choose two clubs from dropdowns, and the model will estimate:
  - The probability that each team is a “strong team”.
  - Which team is the theoretical favorite (who is more likely to win).

The goal is to demonstrate a clean, production-style workflow: data preprocessing, feature engineering, model training with basic overfitting control, and a simple web interface.

---

### Problem Statement

Raw football data is often at the **player level**, while we usually care about **team strength** and **match outcomes**.

This project answers:

> “Given player stats for all teams in the league, can we build a simple model that estimates how strong each team is, and then compare two teams to see who is favored?”

Instead of directly predicting a single match result, we build a **team strength score** (probability of being a “strong” team). This score can then be used to compare any two teams.

---

### Approach & Logic

#### From player-level stats to team-level features

1. **Load player statistics** from a Kaggle CSV (one row per player, with various performance metrics).
2. **Clean the data**:
   - Drop rows with missing `Nationality`, `Age`, or `Jersey Number` (clearly incomplete records).
   - Convert percentage columns (e.g., `Shooting accuracy %`) from strings like `"45%"` to numeric `float` values.
3. **Normalize to per-game metrics**:
   - Remove players with `Appearances == 0` to avoid division by zero.
   - For most numeric counting stats (goals, shots, tackles, etc.), divide by `Appearances` to obtain **per-game** rates.
   - This makes players comparable even if they played different numbers of games.
4. **Filter for stability**:
   - Keep only players with at least **38 appearances** (roughly a full season).
   - This reduces noise from players with very few minutes.
5. **Aggregate to team level (per club)**:
   - Select only numeric columns and keep `Club`.
   - For each `Club`:
     - Take the **mean of all per-game stats** across its players → team-level feature vector.
   - Compute a **team win rate**:
     - `win_rate = (sum of Wins for all players in club) / (sum of Appearances for all players in club)`.
   - Use `win_rate` as the **target** for defining “strong” teams.

#### Target definition

- Compute the **median** of `win_rate` across all clubs.
- Label each club:
  - `1` = “strong team” if `win_rate >= median`.
  - `0` = “not strong team” otherwise.

We then train a binary classifier to predict this label from the aggregated team-level features.

---

### Handling Overfitting

The model uses several simple but effective techniques to reduce overfitting:

- **Feature normalization per game**:
  - Dividing counting stats by appearances (per-game rates) avoids inflated values for players who simply played more time.
- **Filtering for players with ≥ 38 appearances**:
  - Reduces noise and outliers from players with too few games, giving more stable statistics.
- **Aggregation to team-level features**:
  - Instead of learning on very noisy individual player records, the model sees one “summary vector” per club, which is richer and smoother.
- **Train/Test split**:
  - The data is split into training and test sets using `train_test_split` with `stratify=y`.
  - This ensures evaluation on unseen teams and helps detect overfitting.
- **Logistic Regression with L2 regularization**:
  - The classifier is a `LogisticRegression` model wrapped in a `Pipeline`:
    - `StandardScaler` → `LogisticRegression(penalty="l2", C=1.0)`.
  - **L2 regularization** penalizes large weights, encouraging simpler models that generalize better.

---

### Tech Stack

- **Language**: Python
- **Data processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Web app / UI**: Streamlit
- **Visualization / Metrics**: Scikit-Learn (accuracy, classification report)

---

### Project Structure

- `premier_league_team_strength_model.py`  
  - Core data loading, preprocessing, feature engineering, and model training/evaluation logic.
- `streamlit_app.py`  
  - Streamlit user interface:
    - Loads and preprocesses data.
    - Trains (and caches) the model.
    - Provides dropdowns to select two clubs and displays predicted probabilities + favorite.
- `data/premier_league_players.csv` (not included)  
  - Expected location for the Kaggle CSV with player statistics.

---

### Getting Started

#### 1. Clone the repository

git clone https://github.com/<your-username>/premier-league-team-strength-predictor.git
cd premier-league-team-strength-predictor
