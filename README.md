# ⚽ Premier League Match Predictor (v2.0)

A dynamic, end-to-end Machine Learning web application that predicts Premier League team strengths, simulates matchups, and forecasts the season champion using live data.

**🔗 [Live Demo: Play with the App here!](https://premier-league-predictor-gq6jzykutzbajpzy7afvwh.streamlit.app/)**

## 🚀 What's New in v2.0?
The project has evolved from a static model running on local CSV files to a live web application:
* **Live API Integration:** Fetches real-time standings and team data via `football-data.org`.
* **Web Scraping:** Built a custom scraper using `pandas.read_html` and `requests` (with User-Agent spoofing to bypass 403 errors) to extract current "Form" (Last 5 matches) data. Cleaned dirty HTML strings using Regex.
* **Improved Model Accuracy:** Optimized the Random Forest classifier. Adjusted the Train/Test split ratios and utilized `GridSearchCV` for hyperparameter tuning, successfully increasing the model's test accuracy from 80% to 87.5%.
* **Stochastic Monte Carlo Simulator:** Added a "Predict Champion" feature. Instead of deterministic outputs, the simulator uses the model's `predict_proba` as weighted probabilities to simulate matchups between top-tier teams, introducing realistic variance.
* **Advanced Data Viz:** Integrated `Plotly` for interactive Head-to-Head (H2H) statistical comparisons based on live data.
* **Cloud Deployment:** Hosted seamlessly on Streamlit Community Cloud.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV)
* **Data Engineering:** Pandas, NumPy
* **Data Collection:** REST APIs, Web Scraping (`BeautifulSoup`, `lxml`)
* **Frontend & Visualization:** Streamlit, Plotly
* **Deployment:** Streamlit Community Cloud, GitHub

## 📂 Project Structure
* `streamlit_app.py`: The main frontend application and UI logic.
* `premier_league_team_strength_model.py`: The core ML pipeline (data cleaning, training, and evaluation).
* `api_data_fetcher.py`: Handles API requests and the Web Scraping logic.
* `data/`: Contains the historical `premier_league_players.csv` dataset used to train the base model.
* `requirements.txt`: Dependencies for cloud deployment.

## 💻 Run it Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your API key: `FOOTBALL_DATA_ORG_KEY=your_api_key_here`
4. Run the app: `python -m streamlit run streamlit_app.py`
