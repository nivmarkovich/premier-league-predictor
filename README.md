# ⚽ Premier League Predictor & Season Simulator (v3.0)

A dynamic, end-to-end Machine Learning web application that predicts Premier League match outcomes and simulates the final league table using live data, historical stats, and Expected Points (EV) mathematics.

**🔗 [Live Demo: Play with the App here!]([[INSERT_YOUR_APP_LINK_HERE]](https://premier-league-predictor-gq6jzykutzbajpzy7afvwh.streamlit.app/))**

## 🚀 Key Features & The Math Behind the App
This project evolved from a static Random Forest model to a robust, live-updating forecasting engine:

* **Expected Points (EV) Simulator:** Instead of relying on high-variance Monte Carlo simulations ("dice rolls"), the season simulator calculates the precise Expected Points for each future match based on combined probabilities, ensuring a realistic and statistically sound final league table.
* **Smart Probability Blending:** The prediction engine uses a sophisticated weighted approach to reflect true football reality:
  * **Current Form (PPG Power Law):** Heavily weights the current season's Points Per Game to capture momentum.
  * **Head-to-Head (H2H):** Factors in the results of the first-round fixtures between the teams.
  * **Historical ML Model:** Uses a tuned `RandomForestClassifier` (87.5% accuracy) trained on historical Premier League data to establish baseline team strength.
* **Live API Integration:** Fetches real-time standings and remaining scheduled fixtures via `football-data.org`.
* **Advanced Data Engineering & Fallbacks:** * Implemented `difflib` and Hardcoded Manual Mapping to handle stubborn "Name Mismatches" between API endpoints (e.g., *Brentford* vs *Brentford FC*).
  * Built `try...except` fallback mechanisms to generate form-based predictions even for newly promoted teams (Out-of-Vocabulary handling) that the historical ML model doesn't recognize.
* **Modern UI/UX:** A custom-styled Streamlit interface featuring dynamic Head-to-Head probability scoreboards, visual form guides (W/D/L), and color-coded final standings.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV), Expected Value (EV) Mathematics
* **Data Engineering:** Pandas, NumPy, Difflib
* **Data Collection:** REST APIs (`requests`)
* **Frontend & Visualization:** Streamlit, Plotly, Custom CSS Injection
* **Deployment:** Streamlit Community Cloud, GitHub CI/CD

## 📂 Project Structure
* `streamlit_app.py`: The main frontend application, UI styling, and Expected Points simulation loop.
* `premier_league_team_strength_model.py`: The core ML pipeline (data cleaning, training, and model export).
* `api_data_fetcher.py`: Handles API requests, live standings, remaining fixtures, and H2H data.
* `data/`: Contains the historical datasets used to train the base model.
* `requirements.txt`: Dependencies for cloud deployment.

## 💻 Run it Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your API key: `FOOTBALL_DATA_ORG_KEY=your_api_key_here`
4. Run the app: `streamlit run streamlit_app.py`
