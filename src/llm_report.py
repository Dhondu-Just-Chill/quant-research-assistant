# src/llm_report.py

import os
import json
import time
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from google import genai
from dotenv import load_dotenv

from statistical_analysis import run_analysis

load_dotenv()

# Configure Gemini
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL  = "gemini-2.5-flash"


# ── NEWS FETCHING ─────────────────────────────────────────────────────

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """
    Fetch recent news headlines from yfinance.

    yfinance returns a list of article dicts with keys:
    title, publisher, link, providerPublishTime.

    Returns a normalized list of {headline, source, date} dicts
    so both news sources share the same output format.
    """
    print(f"  Fetching yfinance news for {ticker}...")
    try:
        t       = yf.Ticker(ticker)
        articles = t.news or []
        results  = []

        for article in articles:
            # providerPublishTime is a unix timestamp
            pub_date = datetime.fromtimestamp(
                article.get("providerPublishTime", 0)
            ).strftime("%Y-%m-%d")

            results.append({
                "headline": article.get("title", ""),
                "source":   article.get("publisher", "yfinance"),
                "date":     pub_date
            })

        print(f"  Found {len(results)} headlines from yfinance")
        return results

    except Exception as e:
        print(f"  yfinance news failed: {e}")
        return []


def fetch_newsapi_headlines(ticker: str, company_name: str) -> list[dict]:
    """
    Fetch recent headlines from NewsAPI.

    Searches for both ticker symbol and company name to maximize coverage.
    Uses the free tier endpoint (everything) filtered to last 7 days.

    Args:
        ticker       : stock symbol e.g. 'AAPL'
        company_name : full company name e.g. 'Apple' for better search results

    Returns normalized list of {headline, source, date} dicts.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("  NEWSAPI_KEY not found in .env — skipping NewsAPI")
        return []

    print(f"  Fetching NewsAPI headlines for {ticker} / {company_name}...")

    # Search last 7 days
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    query     = f"{ticker} OR {company_name} stock"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": 30,
        "apiKey":   api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if data.get("status") != "ok":
            print(f"  NewsAPI error: {data.get('message')}")
            return []

        results = []
        for article in data.get("articles", []):
            pub_date = article.get("publishedAt", "")[:10]  # extract YYYY-MM-DD
            results.append({
                "headline": article.get("title", ""),
                "source":   article.get("source", {}).get("name", "NewsAPI"),
                "date":     pub_date
            })

        print(f"  Found {len(results)} headlines from NewsAPI")
        return results

    except Exception as e:
        print(f"  NewsAPI request failed: {e}")
        return []


def fetch_all_news(ticker: str, company_name: str) -> list[dict]:
    """
    Fetch and deduplicate headlines from both yfinance and NewsAPI.

    Deduplication is done by normalizing headline text — removes duplicates
    where the same story appears in both sources with slightly different wording.

    Returns combined list sorted by date descending.
    """
    yf_news   = fetch_yfinance_news(ticker)
    api_news  = fetch_newsapi_headlines(ticker, company_name)
    combined  = yf_news + api_news

    # Deduplicate by headline text (case-insensitive, strip whitespace)
    seen      = set()
    unique    = []
    for item in combined:
        key = item["headline"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort by date descending — most recent first
    unique.sort(key=lambda x: x["date"], reverse=True)
    print(f"  Total unique headlines after deduplication: {len(unique)}")
    return unique


# ── SENTIMENT SCORING ─────────────────────────────────────────────────

def score_headlines_with_gemini(headlines: list[dict], ticker: str) -> list[dict]:
    """
    Score each headline's sentiment using Gemini.

    Sends headlines in batches to avoid rate limits and token limits.
    Prompts Gemini to return structured JSON with sentiment and score per headline.

    Sentiment scale:
        score  1.0 : strongly positive
        score  0.0 : neutral
        score -1.0 : strongly negative

    Relevance filter: headlines scored as low relevance to the stock
    are excluded from the aggregate sentiment computation.
    """
    if not headlines:
        return []

    print(f"  Scoring {len(headlines)} headlines with Gemini...")

    # Process in batches of 10 to stay within token limits
    batch_size = 10
    scored     = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]

        # Format headlines for the prompt
        headlines_text = "\n".join([
            f"{j+1}. {h['headline']}"
            for j, h in enumerate(batch)
        ])

        prompt = f"""You are a financial sentiment analyst. Score each headline's sentiment toward {ticker} stock.

Headlines:
{headlines_text}

Return ONLY a JSON array with no markdown, no backticks, no preamble. Each element must have:
- "index": headline number (1-based)
- "sentiment": one of "positive", "negative", "neutral"
- "score": float from -1.0 (very negative) to 1.0 (very positive)
- "relevance": one of "high", "medium", "low"

Example format:
[{{"index": 1, "sentiment": "positive", "score": 0.7, "relevance": "high"}}]"""

        try:
            response = gemini_client.models.generate_content(model=GEMINI_MODEL,contents=prompt)
            text = response.text.strip()

            # Strip markdown code fences if Gemini adds them anyway
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            batch_scores = json.loads(text)

            for score_item in batch_scores:
                idx = score_item["index"] - 1
                if 0 <= idx < len(batch):
                    scored.append({
                        **batch[idx],
                        "sentiment": score_item.get("sentiment", "neutral"),
                        "score":     score_item.get("score", 0.0),
                        "relevance": score_item.get("relevance", "medium")
                    })

            # Respect Gemini free tier rate limit
            time.sleep(1)

        except Exception as e:
            print(f"  Gemini scoring error on batch {i//batch_size + 1}: {e}")
            # Add unscored headlines as neutral rather than dropping them
            for item in batch:
                scored.append({**item, "sentiment": "neutral", "score": 0.0, "relevance": "low"})

    return scored


def compute_sentiment_summary(scored_headlines: list[dict]) -> dict:
    """
    Aggregate individual headline scores into a summary sentiment profile.

    Only high and medium relevance headlines are included in score computation
    to reduce noise from tangentially related articles.

    Returns:
        overall_sentiment  : 'positive', 'negative', or 'neutral'
        average_score      : float -1.0 to 1.0
        headline_count     : total headlines analyzed
        positive_count     : number of positive headlines
        negative_count     : number of negative headlines
        neutral_count      : number of neutral headlines
        notable_headlines  : top 3 most impactful headlines by abs(score)
    """
    if not scored_headlines:
        return {
            "overall_sentiment": "neutral",
            "average_score":     0.0,
            "headline_count":    0,
            "positive_count":    0,
            "negative_count":    0,
            "neutral_count":     0,
            "notable_headlines": []
        }

    # Filter to relevant headlines only
    relevant = [
        h for h in scored_headlines
        if h.get("relevance") in ["high", "medium"]
    ]

    if not relevant:
        relevant = scored_headlines  # fallback — use all if none flagged relevant

    scores         = [h["score"] for h in relevant]
    average_score  = np.mean(scores)

    # Determine overall sentiment from average
    if average_score > 0.1:
        overall = "positive"
    elif average_score < -0.1:
        overall = "negative"
    else:
        overall = "neutral"

    # Top 3 most impactful by absolute score value
    notable = sorted(relevant, key=lambda x: abs(x["score"]), reverse=True)[:3]

    return {
        "overall_sentiment": overall,
        "average_score":     round(float(average_score), 4),
        "headline_count":    len(scored_headlines),
        "positive_count":    sum(1 for h in relevant if h["sentiment"] == "positive"),
        "negative_count":    sum(1 for h in relevant if h["sentiment"] == "negative"),
        "neutral_count":     sum(1 for h in relevant if h["sentiment"] == "neutral"),
        "notable_headlines": [h["headline"] for h in notable]
    }


# ── REPORT GENERATION ─────────────────────────────────────────────────

def load_ml_prediction(ticker: str, data_path: str = None) -> dict:
    """
    Load saved model and generate a prediction on the most recent data.

    Loads the model + feature list saved by ml_forecasting.py.
    Fetches the latest data, engineers features, and runs inference
    on the most recent row — today's market state.

    Returns prediction direction, probability, and feature snapshot.
    """
    model_path = f"models/{ticker}_model.pkl"
    if not os.path.exists(model_path):
        return {"error": f"No model found for {ticker}. Run ml_forecasting.py first."}

    saved    = joblib.load(model_path)
    ml_model = saved["model"]
    features = saved["features"]

    # Load raw data and engineer features
    raw_path = data_path or f"data/{ticker}_raw.csv"
    if not os.path.exists(raw_path):
        return {"error": f"No data found at {raw_path}"}

    # Import feature engineering from ml_forecasting
    from ml_forecasting import add_technical_indicators

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df, ticker)

    # Get the most recent row — today's feature state
    latest   = df[features].iloc[-1:]
    prob     = ml_model.predict_proba(latest)[0]

    # prob[1] = probability of Up, prob[0] = probability of Down
    direction   = "Up" if prob[1] > 0.5 else "Down"
    confidence  = max(prob[0], prob[1])

    return {
        "direction":       direction,
        "confidence":      round(float(confidence) * 100, 1),
        "prob_up":         round(float(prob[1]) * 100, 1),
        "prob_down":       round(float(prob[0]) * 100, 1),
        "features_used":   len(features),
        "prediction_date": df.index[-1].strftime("%Y-%m-%d")
    }


def generate_research_report(
    ticker:        str,
    company_name:  str,
    stats:         dict,
    ml_prediction: dict,
    sentiment:     dict
) -> dict:
    """
    Generate a structured JSON research report using Gemini.

    Assembles all analysis layers into a single prompt and asks Gemini
    to synthesize them into a professional research report with
    clearly defined sections.

    The prompt explicitly instructs Gemini to return only JSON —
    no markdown fences, no preamble — so the output can be parsed
    directly without cleanup.

    Args:
        ticker       : stock symbol
        company_name : full company name
        stats        : output from statistical_analysis.run_analysis()
        ml_prediction: output from load_ml_prediction()
        sentiment    : output from compute_sentiment_summary()

    Returns structured dict with report sections.
    """
    prompt = f"""You are a senior quantitative analyst writing a research report.
Analyze the following data and generate a structured JSON research report.

STOCK: {company_name} ({ticker})
DATE: {datetime.now().strftime("%Y-%m-%d")}

STATISTICAL PROFILE:
- Annualized Return: {stats.get('annualized_return')}%
- Annualized Volatility: {stats.get('annualized_volatility')}%
- Sharpe Ratio: {stats.get('sharpe_ratio')}
- Max Drawdown: {stats.get('max_drawdown')}%
- VaR 95% (1-day): {stats.get('var_95_1day')}%
- Skewness: {stats.get('skewness')}
- Excess Kurtosis: {stats.get('excess_kurtosis')}
- Returns Normal: {stats.get('returns_are_normal')}

ML FORECAST:
- Direction: {ml_prediction.get('direction')}
- Confidence: {ml_prediction.get('confidence')}%
- Probability Up: {ml_prediction.get('prob_up')}%
- Probability Down: {ml_prediction.get('prob_down')}%
- Features Used: {ml_prediction.get('features_used')}

NEWS SENTIMENT:
- Overall: {sentiment.get('overall_sentiment')}
- Score: {sentiment.get('average_score')} (scale: -1.0 to 1.0)
- Headlines Analyzed: {sentiment.get('headline_count')}
- Positive: {sentiment.get('positive_count')} | Negative: {sentiment.get('negative_count')} | Neutral: {sentiment.get('neutral_count')}
- Notable Headlines: {sentiment.get('notable_headlines')}

Return ONLY a JSON object with no markdown, no backticks, no preamble:
{{
    "ticker": "{ticker}",
    "company": "{company_name}",
    "report_date": "{datetime.now().strftime('%Y-%m-%d')}",
    "executive_summary": "2-3 sentence synthesis of the overall picture",
    "statistical_profile": {{
        "risk_rating": "Low | Medium | High | Very High",
        "key_metrics_interpretation": "plain English interpretation of the stats",
        "volatility_context": "what the volatility level means for an investor"
    }},
    "ml_forecast": {{
        "direction": "Up | Down",
        "confidence": "percentage as string",
        "reasoning": "what technical and macro factors drive this prediction",
        "reliability_note": "honest assessment of model limitations"
    }},
    "sentiment_analysis": {{
        "overall": "positive | negative | neutral",
        "score": 0.0,
        "market_narrative": "what the news flow suggests about market perception",
        "key_themes": "main themes across the headlines",
        "notable_headlines": []
    }},
    "risk_factors": "key risks an investor should be aware of",
    "investment_thesis": "balanced view — bull case and bear case in 2-3 sentences each",
    "disclaimer": "This report is generated by an automated system for educational purposes only and does not constitute financial advice."
}}"""

    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL,contents=prompt)
        text = response.text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]

        report = json.loads(text.strip())
        return report

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw response: {text[:500]}")
        return {"error": "Failed to parse Gemini response as JSON", "raw": text}

    except Exception as e:
        print(f"  Report generation failed: {e}")
        return {"error": str(e)}


# ── COMPANY NAME LOOKUP ───────────────────────────────────────────────

def get_company_name(ticker: str) -> str:
    """
    Fetch the full company name for a ticker using yfinance.
    Falls back to ticker symbol if unavailable.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


# ── MAIN PIPELINE ─────────────────────────────────────────────────────

def run_llm_pipeline(ticker: str) -> dict:
    """
    Full Week 3 pipeline for a given ticker.

    Stages:
        1. Fetch company name
        2. Fetch news from yfinance + NewsAPI
        3. Score headlines with Gemini
        4. Compute sentiment summary
        5. Load ML model and generate prediction
        6. Run statistical analysis
        7. Generate structured research report with Gemini
        8. Save report to outputs/

    Returns the complete report dict.
    """
    print(f"\n{'='*50}")
    print(f"Running LLM pipeline for {ticker}")
    print(f"{'='*50}")

    # Stage 1 — company name
    company_name = get_company_name(ticker)
    print(f"Company: {company_name}")

    # Stage 2 — fetch news
    print("\n[1/5] Fetching news...")
    headlines = fetch_all_news(ticker, company_name)

    # Stage 3 — score sentiment
    print("\n[2/5] Scoring sentiment...")
    scored     = score_headlines_with_gemini(headlines, ticker)
    sentiment  = compute_sentiment_summary(scored)
    print(f"  Sentiment: {sentiment['overall_sentiment']} (score: {sentiment['average_score']})")

    # Stage 4 — ML prediction
    print("\n[3/5] Loading ML prediction...")
    ml_pred = load_ml_prediction(ticker)
    if "error" in ml_pred:
        print(f"  Warning: {ml_pred['error']}")
    else:
        print(f"  Prediction: {ml_pred['direction']} ({ml_pred['confidence']}% confidence)")

    # Stage 5 — statistical analysis
    print("\n[4/5] Running statistical analysis...")
    stats = run_analysis(ticker)

    # Stage 6 — generate report
    print("\n[5/5] Generating research report...")
    report = generate_research_report(ticker, company_name, stats, ml_pred, sentiment)

    # Stage 7 — save report
    os.makedirs("outputs", exist_ok=True)
    report_path = f"outputs/{ticker}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    return report


if __name__ == "__main__":
    for ticker in ["AAPL", "SPY", "GOOGL", "AMZN", "MSFT"]:
        report = run_llm_pipeline(ticker)

        if "error" not in report:
            print(f"\n{ticker} Report Summary:")
            print(f"  Executive Summary: {report.get('executive_summary', '')[:200]}...")
            print(f"  ML Forecast:       {report.get('ml_forecast', {}).get('direction')} - {report.get('ml_forecast', {}).get('confidence')}")
            print(f"  Sentiment:         {report.get('sentiment_analysis', {}).get('overall')}")
        else:
            print(f"\n{ticker} report failed: {report.get('error')}")
        print("---")
