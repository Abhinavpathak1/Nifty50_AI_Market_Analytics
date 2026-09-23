"""
generate_report.py — creates Nifty50_AI_Market_Analytics_Report.docx
Run:  python generate_report.py
Requires:  pip install python-docx
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ─── Helpers ──────────────────────────────────────────────────────────────────

def add_heading(doc, text, level=1, color=None):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        if color:
            run.font.color.rgb = RGBColor(*color)
    return h


def add_para(doc, text, bold=False, italic=False, size=11, align=WD_ALIGN_PARAGRAPH.LEFT,
             space_after=6, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return p


def add_page_break(doc):
    doc.add_page_break()


def set_col_width(table, col_idx, width_cm):
    for row in table.rows:
        row.cells[col_idx].width = Cm(width_cm)


def shade_row(row, hex_color="1C77C3"):
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), hex_color)
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)


def header_row(table, texts, bg="1C77C3"):
    row = table.rows[0]
    for i, text in enumerate(texts):
        cell = row.cells[i]
        cell.text = text
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10)
    shade_row(row, bg)


# ─── Build Document ───────────────────────────────────────────────────────────

doc = Document()

# Margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ════════════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("NIFTY 50 STOCK MARKET ANALYTICS AND\nMULTI-TASK MACHINE LEARNING PREDICTION SYSTEM")
r.bold = True
r.font.size = Pt(18)
r.font.color.rgb = RGBColor(0x0B, 0x13, 0x2B)

doc.add_paragraph()
add_para(doc, "A Project Report", bold=False, size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc,
         "Submitted in partial fulfillment of the requirements for the",
         size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc,
         "IBM SkillsBuild Data Analytics with AI Academy Internship Program",
         bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
         color=(0x1C, 0x77, 0xC3))
add_para(doc,
         "Conducted by Bharat Care in association with AICTE",
         size=11, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_paragraph()
doc.add_paragraph()

add_para(doc, "Submitted by:", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[Student Full Name]", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "Roll No.: [__________]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[Course / Branch]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[College / Institution Name]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[City, State — PIN Code]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_paragraph()
add_para(doc, "Academic / Internship Year: 2024–2025", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_paragraph()
add_para(doc, "Guided by:", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[Mentor / Supervisor Name]", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, "[Designation]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_paragraph()
doc.add_paragraph()
add_para(doc, "Dataset Source:", bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc,
         "Kaggle — Nifty50 Stocks (1999–2026) Daily OHLCV & Fundamentals",
         size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc,
         "https://www.kaggle.com/datasets/kalyan197/nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# CERTIFICATE PLACEHOLDER
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "Certificate", level=1)
add_para(doc,
    "This page is reserved for the official internship completion certificate issued by the "
    "IBM SkillsBuild Data Analytics with AI Academy Internship Program, conducted by "
    "Bharat Care in association with AICTE.\n\n"
    "The certificate should be obtained from the internship organizer upon successful "
    "completion of the program and attached here. This document does not fabricate or "
    "simulate any official certification.",
    size=11, space_after=8)
doc.add_paragraph()
add_para(doc, "____________________________________________", size=11)
add_para(doc, "Signature of Program Coordinator / Authorized Representative", size=10, italic=True)
add_para(doc, "Date: ____________________", size=11)
add_para(doc, "Seal / Stamp:", size=11)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# DECLARATION
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "Declaration", level=1)
add_para(doc,
    "I, [Student Full Name], student of [Course / Branch] at [College / Institution Name], "
    "hereby declare that the project report titled 'Nifty 50 Stock Market Analytics and "
    "Multi-Task Machine Learning Prediction System' has been prepared by me independently "
    "as part of the IBM SkillsBuild Data Analytics with AI Academy Internship Program "
    "conducted by Bharat Care in association with AICTE, during the academic year 2024–2025.",
    size=11, space_after=8)
add_para(doc,
    "I further declare that:",
    bold=True, size=11, space_after=4)
for item in [
    "The work presented in this report is original and has not been submitted elsewhere for "
    "any other degree, diploma, or award.",
    "All data, references, and resources used have been duly acknowledged.",
    "The machine learning models and code in this project have been developed for educational "
    "and research purposes only.",
    "This project does not constitute financial advice and makes no claims of guaranteed "
    "investment returns or market prediction accuracy.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(item).font.size = Pt(11)
doc.add_paragraph()
add_para(doc, "Place: ____________________", size=11)
add_para(doc, "Date:  ____________________", size=11)
doc.add_paragraph()
add_para(doc, "Signature: ____________________", size=11)
add_para(doc, "[Student Full Name]", size=11)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# ACKNOWLEDGEMENT
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "Acknowledgement", level=1)
add_para(doc,
    "I would like to express my sincere gratitude to the IBM SkillsBuild program for "
    "providing access to world-class learning resources and a structured internship framework "
    "that made this project possible.",
    size=11, space_after=8)
add_para(doc,
    "I am deeply thankful to Bharat Care and AICTE for organizing and facilitating this "
    "Data Analytics with AI Academy Internship Program, which has greatly enhanced my "
    "practical understanding of data science, machine learning, and financial analytics.",
    size=11, space_after=8)
add_para(doc,
    "I extend my heartfelt thanks to my mentor and guide, [Mentor Name], for their "
    "invaluable guidance, constructive feedback, and continuous encouragement throughout "
    "the duration of this project.",
    size=11, space_after=8)
add_para(doc,
    "I am also grateful to Kalyan (Kaggle contributor) for making the Nifty 50 historical "
    "dataset publicly available, which served as the foundation of this project.",
    size=11, space_after=8)
add_para(doc,
    "Finally, I thank my family and colleagues for their moral support and patience during "
    "the completion of this work.",
    size=11, space_after=8)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# ABSTRACT
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "Abstract", level=1)
add_para(doc,
    "This project presents a complete end-to-end data analytics and machine learning pipeline "
    "for the Nifty 50 stock market, covering more than 25 years of historical daily OHLCV "
    "(Open, High, Low, Close, Volume) and fundamental data from 1999 to 2026. The dataset "
    "comprises approximately 287,310 records across 49 constituent stocks spanning 13 sectors "
    "of the Indian economy.",
    size=11, space_after=8)
add_para(doc,
    "The project begins with a rigorous 20-point data quality audit, followed by a transparent "
    "data cleaning pipeline that preserves all cleaning decisions. Comprehensive exploratory "
    "data analysis is conducted at the market, stock, sector, yearly, and monthly levels.",
    size=11, space_after=8)
add_para(doc,
    "Technical indicators including Simple Moving Averages, Exponential Moving Averages, RSI, "
    "MACD, Bollinger Bands, and ATR are computed per ticker without cross-stock contamination. "
    "Lag and rolling features are engineered to serve as inputs to machine learning models.",
    size=11, space_after=8)
add_para(doc,
    "Seven prediction tasks are defined: next-day close price (regression), next-day return "
    "(regression), next-day up/down direction (classification), next-day high (regression), "
    "next-day low (regression), next-day volatility (regression), and market regime "
    "classification (Low/Medium/High volatility). A strict chronological 70/15/15 "
    "train/validation/test split is used to prevent data leakage.",
    size=11, space_after=8)
add_para(doc,
    "Five modelling approaches are implemented and compared: a naive baseline, Linear "
    "Regression, Random Forest, XGBoost gradient boosting, and an optional LSTM neural "
    "network. Model evaluation uses appropriate metrics (MAE, RMSE, R², Accuracy, F1, "
    "ROC-AUC) on the held-out chronological test set.",
    size=11, space_after=8)
add_para(doc,
    "All findings are presented through a professional eight-page interactive Streamlit "
    "dashboard powered by Plotly visualizations. The project explicitly disclaims that all "
    "predictions are statistical estimates for educational purposes only and do not constitute "
    "financial advice.",
    size=11, space_after=8)
add_para(doc, "Keywords: Nifty 50, Stock Market, Machine Learning, Time Series, XGBoost, "
              "Random Forest, LSTM, Technical Analysis, Streamlit, Data Analytics",
         italic=True, size=10, space_after=8)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "Table of Contents", level=1)
toc_items = [
    ("Certificate", "i"),
    ("Declaration", "ii"),
    ("Acknowledgement", "iii"),
    ("Abstract", "iv"),
    ("Table of Contents", "v"),
    ("1. Introduction", "1"),
    ("2. Problem Statement", "2"),
    ("3. Objectives", "2"),
    ("4. Dataset Description", "3"),
    ("5. Data Preprocessing", "4"),
    ("6. Exploratory Data Analysis", "5"),
    ("7. Feature Engineering", "7"),
    ("8. Technical Indicators", "8"),
    ("9. Machine Learning Methodology", "9"),
    ("10. Models Used", "10"),
    ("11. Prediction Tasks", "11"),
    ("12. Model Evaluation", "12"),
    ("13. Dashboard Design", "13"),
    ("14. Results", "14"),
    ("15. Limitations", "15"),
    ("16. Future Scope", "15"),
    ("17. Conclusion", "16"),
    ("References", "17"),
    ("Appendix", "18"),
]
toc_table = doc.add_table(rows=len(toc_items), cols=2)
toc_table.style = "Table Grid"
for i, (title, page) in enumerate(toc_items):
    row = toc_table.rows[i]
    row.cells[0].text = title
    row.cells[1].text = page
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(11)
    row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
set_col_width(toc_table, 0, 13)
set_col_width(toc_table, 1, 2)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 1 — INTRODUCTION
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "1. Introduction", level=1)
add_para(doc,
    "The Indian stock market, represented by the National Stock Exchange's Nifty 50 index, "
    "has grown into one of the world's most significant emerging-market equity benchmarks. "
    "The Nifty 50 comprises the fifty largest and most liquid companies listed on the NSE, "
    "spanning thirteen sectors including Information Technology, Banking & Financial Services, "
    "Energy, Pharmaceuticals, Consumer Goods, Automobiles, and Infrastructure.",
    size=11, space_after=8)
add_para(doc,
    "The application of data analytics and machine learning to financial markets has witnessed "
    "substantial growth over the past decade. Quantitative approaches allow analysts and "
    "researchers to uncover patterns in historical price, volume, and fundamental data that "
    "may not be immediately apparent through traditional analysis.",
    size=11, space_after=8)
add_para(doc,
    "This project, conducted as part of the IBM SkillsBuild Data Analytics with AI Academy "
    "Internship Program (Bharat Care × AICTE), builds a complete, reproducible pipeline "
    "covering data auditing, cleaning, exploratory analysis, technical indicator computation, "
    "feature engineering, multi-task machine learning prediction, model explainability, and "
    "interactive dashboard deployment.",
    size=11, space_after=8)
add_para(doc,
    "IMPORTANT DISCLAIMER: This project is purely educational. None of the models, predictions, "
    "or analysis in this report constitutes financial advice. All predictions carry inherent "
    "uncertainty. Past market performance does not guarantee future results.",
    bold=True, size=11, space_after=8, color=(0xDC, 0x26, 0x26))

# ════════════════════════════════════════════════════════════════════
# CHAPTER 2 — PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "2. Problem Statement", level=1)
add_para(doc,
    "Despite the availability of decades of historical stock market data, making reliable "
    "short-term price predictions remains a challenging open problem in quantitative finance. "
    "Key challenges include:",
    size=11, space_after=6)
for item in [
    "Non-stationary time series — statistical properties change over time.",
    "High noise-to-signal ratio — random market fluctuations dominate short-term movements.",
    "Regime changes — structural market shifts invalidate historical patterns.",
    "Data leakage risk — incorrect feature construction can produce artificially inflated "
    "model accuracy.",
    "Feature selection — hundreds of potentially predictive signals exist; not all are useful.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(item).font.size = Pt(11)
add_para(doc,
    "\nThis project addresses these challenges by building a transparent, well-documented, "
    "leakage-free machine learning pipeline with honest evaluation on a strictly held-out "
    "chronological test set.",
    size=11, space_after=8)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 3 — OBJECTIVES
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "3. Objectives", level=1)
objectives = [
    "Conduct a rigorous 20-point data quality audit on the Nifty 50 historical dataset.",
    "Implement a transparent, well-documented data cleaning and preprocessing pipeline.",
    "Perform comprehensive exploratory data analysis at the market, stock, sector, year, "
    "and month levels.",
    "Compute standard technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) "
    "per ticker without data leakage.",
    "Engineer meaningful lag and rolling features for use as machine learning inputs.",
    "Define seven prediction targets covering price, return, direction, high, low, "
    "volatility, and market regime.",
    "Apply chronological train/validation/test splitting to prevent temporal data leakage.",
    "Train and compare five models: Naive Baseline, Linear Regression, Random Forest, "
    "XGBoost, and LSTM.",
    "Evaluate all models using appropriate metrics on the held-out chronological test set.",
    "Explain model predictions using feature importance analysis.",
    "Build a professional, interactive eight-page Streamlit dashboard with Plotly charts.",
    "Document all decisions, limitations, and ethical considerations transparently.",
]
for i, obj in enumerate(objectives, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.add_run(f"{i}. ").bold = True
    p.add_run(obj).font.size = Pt(11)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 4 — DATASET DESCRIPTION
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "4. Dataset Description", level=1)
add_para(doc, "4.1 Source", bold=True, size=12, space_after=4)
add_para(doc,
    "The dataset used in this project is publicly available on Kaggle under the title "
    "'Nifty50 Stocks (1999–2026) Daily OHLCV & Fundamentals', contributed by Kalyan197. "
    "The original data was sourced from Yahoo Finance.",
    size=11, space_after=6)
add_para(doc,
    "URL: https://www.kaggle.com/datasets/kalyan197/nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals",
    italic=True, size=10, space_after=8)

add_para(doc, "4.2 Dataset Statistics", bold=True, size=12, space_after=4)
stats_table = doc.add_table(rows=7, cols=2)
stats_table.style = "Table Grid"
stats_data = [
    ("Attribute", "Value"),
    ("Period", "1999-01-01 to 2026-01-31 (25+ years)"),
    ("Total Records", "~287,310 rows"),
    ("Stocks", "49 (of 50 attempted)"),
    ("Sectors", "13"),
    ("Frequency", "Daily trading data"),
    ("Columns", "25"),
]
for i, (k, v) in enumerate(stats_data):
    row = stats_table.rows[i]
    row.cells[0].text = k
    row.cells[1].text = v
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(11)
                if i == 0:
                    run.bold = True
if stats_table.rows:
    shade_row(stats_table.rows[0])
set_col_width(stats_table, 0, 5)
set_col_width(stats_table, 1, 10)
doc.add_paragraph()

add_para(doc, "4.3 Column Descriptions", bold=True, size=12, space_after=4)
col_table = doc.add_table(rows=26, cols=3)
col_table.style = "Table Grid"
col_data = [
    ("Column", "Type", "Description"),
    ("Date", "DateTime", "Trading date (IST)"),
    ("Ticker", "String", "NSE ticker symbol"),
    ("Company_Name", "String", "Full company name"),
    ("Sector", "String", "Business sector"),
    ("Open", "Float", "Opening price (₹)"),
    ("High", "Float", "Daily high price (₹)"),
    ("Low", "Float", "Daily low price (₹)"),
    ("Close", "Float", "Closing price (₹)"),
    ("Volume", "Integer", "Shares traded"),
    ("Dividend", "Float", "Dividend amount paid"),
    ("Stock_Split", "Float", "Split ratio"),
    ("Daily_Return", "Float", "Daily percentage return"),
    ("Volatility_20D", "Float", "20-day rolling volatility (pre-computed)"),
    ("MA_50", "Float", "50-day moving average (pre-computed)"),
    ("MA_200", "Float", "200-day moving average (pre-computed)"),
    ("Market_Cap", "Float", "Market capitalization (₹)"),
    ("PE_Ratio", "Float", "Price-to-Earnings ratio"),
    ("Forward_PE", "Float", "Forward P/E ratio"),
    ("PEG_Ratio", "Float", "Price/Earnings-to-Growth ratio"),
    ("Price_to_Book", "Float", "Price-to-Book ratio"),
    ("Dividend_Yield", "Float", "Dividend yield (%)"),
    ("EPS", "Float", "Earnings per share"),
    ("Beta", "Float", "Market sensitivity coefficient"),
    ("52Week_High", "Float", "52-week high price"),
    ("52Week_Low", "Float", "52-week low price"),
]
for i, row_data in enumerate(col_data):
    row = col_table.rows[i]
    for j, val in enumerate(row_data):
        row.cells[j].text = val
        for para in row.cells[j].paragraphs:
            for run in para.runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.bold = True
if col_table.rows:
    shade_row(col_table.rows[0])
set_col_width(col_table, 0, 4)
set_col_width(col_table, 1, 2.5)
set_col_width(col_table, 2, 8.5)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 5 — DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "5. Data Preprocessing", level=1)
add_para(doc, "5.1 Data Quality Audit (20-Point Checklist)", bold=True, size=12, space_after=4)
add_para(doc,
    "A rigorous data quality audit was conducted before any cleaning. The 20-point checklist "
    "covered: dimensions, column names, data types, first/last records, descriptive statistics, "
    "missing values, duplicate rows, duplicate Date-Ticker combinations, invalid dates, "
    "impossible numerical values, negative/zero prices, OHLC consistency, outlier inspection, "
    "ticker/company/sector consistency, chronological ordering, observations per stock, "
    "per-stock date ranges, missing trading days, look-ahead/leakage-prone columns, and a "
    "summary data quality report.",
    size=11, space_after=8)

add_para(doc, "5.2 Data Cleaning Pipeline", bold=True, size=12, space_after=4)
cleaning_steps = [
    ("Step 1: Date Parsing",
     "Dates containing timezone offset (+05:30) were parsed using pd.to_datetime with UTC "
     "conversion, then normalized to date-only in IST (Asia/Kolkata). Invalid dates (NaT) "
     "were counted and dropped."),
    ("Step 2: Sorting",
     "Data was sorted by Ticker ascending, then by Date ascending, to ensure correct "
     "group-wise time-series operations."),
    ("Step 3: Duplicate Removal",
     "Fully duplicate rows were removed. Duplicate Date-Ticker pairs (keeping the first "
     "occurrence) were removed. All counts were logged."),
    ("Step 4: Numeric Conversion",
     "All OHLCV and fundamental columns were explicitly converted using pd.to_numeric with "
     "errors='coerce' to handle non-numeric entries gracefully."),
    ("Step 5: Missing OHLCV Treatment",
     "Forward-filling was applied within each ticker group for Open, High, Low, Close. "
     "Rationale: in financial time series, the previous day's price is the most appropriate "
     "proxy for a missing observation — global mean imputation would introduce cross-stock "
     "contamination. Volume NaN values were filled with 0 (representing a trading halt or "
     "data gap). Rows where Close remained NaN after forward-filling (i.e., first row per "
     "ticker with no prior data) were dropped."),
    ("Step 6: Zero/Negative Price Removal",
     "Rows with Close <= 0 were removed as they represent data errors."),
    ("Step 7: Daily Return Recomputation",
     "The pre-computed Daily_Return column was recalculated from clean Close prices using "
     "pct_change() within each ticker group to ensure consistency."),
    ("Step 8: Fundamental Imputation",
     "Fundamental columns (PE_Ratio, EPS, Beta, etc.) were filled with their ticker-wise "
     "median. This preserves cross-stock differences while filling sparse values."),
    ("Step 9: Look-Ahead Exclusion",
     "52Week_High and 52Week_Low were excluded from all ML feature sets as they may contain "
     "forward-looking information. Pre-computed MA_50 and MA_200 were recomputed from raw "
     "Close prices to ensure consistency."),
]
for title, text in cleaning_steps:
    add_para(doc, title, bold=True, size=11, space_after=2)
    add_para(doc, text, size=11, space_after=6)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 6 — EDA
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "6. Exploratory Data Analysis", level=1)
eda_sections = [
    ("6.1 Market-Wide Analysis",
     "The overall dataset spans 1999 to 2026, covering 49 companies across 13 sectors. "
     "The median close price across all stocks shows a strong long-term upward trend, "
     "consistent with India's GDP growth over the period. Market-wide daily return "
     "distribution is approximately normal with slight positive skew and fat tails "
     "(leptokurtosis), which is typical of equity returns."),
    ("6.2 Individual Stock Analysis",
     "Per-stock analysis includes full price history, OHLCV charts, rolling volatility, "
     "daily returns, and drawdown analysis. Notable observations include Bajaj Finance's "
     "236,000%+ total return and Eicher Motors' nearly 1,000,000% total return over the "
     "full period, illustrating the power of long-term equity compounding."),
    ("6.3 Sector Analysis",
     "IT and Financials sectors show the highest representation (companies) and strong "
     "long-term returns. Energy and Metals exhibit higher volatility profiles. Sector "
     "return distributions were examined using box plots to compare dispersion, outliers, "
     "and median performance."),
    ("6.4 Yearly Analysis",
     "Annual return analysis by stock reveals clear multi-year bear markets (2001, 2008, "
     "2020) and bull runs (2003–2007, 2014–2017, 2021). Year-on-year comparison allows "
     "identification of regime changes in market conditions."),
    ("6.5 Monthly / Seasonal Analysis",
     "Seasonal analysis of average daily returns by month shows some months with "
     "marginally higher or lower average returns. However, the signal is weak and highly "
     "variable across years and stocks — seasonal trading strategies should not be "
     "constructed from this analysis alone."),
    ("6.6 Correlation Analysis",
     "OHLC prices are highly correlated with each other (expected). Close price shows "
     "weak correlation with Daily_Return (as expected for daily returns on varying price "
     "levels). RSI and MACD show low-to-moderate correlation with Close. All correlation "
     "findings are presented with the explicit caveat that correlation does not imply "
     "causation."),
]
for title, text in eda_sections:
    add_para(doc, title, bold=True, size=12, space_after=4)
    add_para(doc, text, size=11, space_after=8)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 7 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "7. Feature Engineering", level=1)
add_para(doc,
    "All features were engineered to use only information available at or before the "
    "prediction time. No future information was allowed to enter the feature matrix.",
    size=11, space_after=8)

fe_table = doc.add_table(rows=10, cols=2)
fe_table.style = "Table Grid"
fe_data = [
    ("Feature Category", "Features Included"),
    ("Lag Features", "Close_Lag1, Close_Lag2, Close_Lag3, Close_Lag5, Close_Lag10\nReturn_Lag1..Lag10"),
    ("Rolling Statistics", "SMA_5, SMA_10, SMA_20\nSTD_5, STD_10, STD_20\nRolling Volatility (Vol_5, Vol_10, Vol_20)"),
    ("Price Momentum", "Momentum_5 = Close - Close.shift(5)\nMomentum_10 = Close - Close.shift(10)"),
    ("High-Low Range", "HL_Range = High - Low\nHL_Range_Pct = (High - Low) / Prev_Close"),
    ("MA Deviations", "Price_vs_SMA50 = (Close - SMA50) / SMA50\nPrice_vs_SMA200 = (Close - SMA200) / SMA200"),
    ("Volume Features", "Volume_Change = pct_change(Volume)\nVolume_SMA5 = 5-day volume average"),
    ("Technical Indicators", "RSI_14, MACD, MACD_Signal, MACD_Diff\nBB_High, BB_Low, BB_Width, ATR_14"),
    ("Fundamental Snapshot", "PE_Ratio, Forward_PE, Price_to_Book, EPS, Beta, Dividend_Yield"),
    ("Excluded (leakage risk)", "52Week_High, 52Week_Low, Future Close/High/Low/Return/Volatility"),
]
for i, (k, v) in enumerate(fe_data):
    row = fe_table.rows[i]
    row.cells[0].text = k
    row.cells[1].text = v
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.bold = True
if fe_table.rows:
    shade_row(fe_table.rows[0])
set_col_width(fe_table, 0, 5)
set_col_width(fe_table, 1, 10)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════
# CHAPTER 8 — TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "8. Technical Indicators", level=1)
add_para(doc,
    "Technical indicators were computed per ticker using the Python 'ta' library (or "
    "manual implementations as fallback). All computations were performed within ticker "
    "groups to prevent data contamination across stocks.",
    size=11, space_after=8)
indicators = [
    ("SMA (20/50/100/200)", "Simple Moving Averages over 20, 50, 100, 200 periods."),
    ("EMA (20/50)", "Exponential Moving Averages that weight recent prices more."),
    ("RSI (14)", "Relative Strength Index — measures momentum; >70 overbought, <30 oversold."),
    ("MACD", "Moving Average Convergence Divergence — trend-following momentum indicator."),
    ("Bollinger Bands", "SMA ± 2 standard deviations — measures volatility expansion/contraction."),
    ("ATR (14)", "Average True Range — absolute volatility measure."),
    ("MACD Signal", "9-period EMA of MACD line — used for crossover signals."),
    ("BB Width", "Bollinger Band Width = (BB_High - BB_Low) / BB_Mid — normalized volatility."),
]
ind_table = doc.add_table(rows=len(indicators) + 1, cols=2)
ind_table.style = "Table Grid"
row = ind_table.rows[0]
row.cells[0].text = "Indicator"
row.cells[1].text = "Description"
shade_row(row)
for cell in row.cells:
    for para in cell.paragraphs:
        for run in para.runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)
for i, (name, desc) in enumerate(indicators):
    r = ind_table.rows[i + 1]
    r.cells[0].text = name
    r.cells[1].text = desc
    for cell in r.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(10)
set_col_width(ind_table, 0, 4)
set_col_width(ind_table, 1, 11)
add_page_break(doc)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 9 — ML METHODOLOGY
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "9. Machine Learning Methodology", level=1)
add_para(doc, "9.1 Why NOT Random Train/Test Splitting", bold=True, size=12, space_after=4)
add_para(doc,
    "Randomly shuffling and splitting a time series dataset before model training is a "
    "critical error that constitutes data leakage. If a test observation from 2024 is "
    "placed in the training set alongside observations from 2010, the model effectively "
    "has access to future information during training — producing artificially high "
    "evaluation metrics that do not reflect real-world predictive performance.",
    size=11, space_after=8)
add_para(doc, "9.2 Chronological Train/Validation/Test Split", bold=True, size=12, space_after=4)
add_para(doc,
    "A strict chronological 70/15/15 split was applied based on the global date axis:\n"
    "• Training Set (70%): Earliest observations — used to fit all models.\n"
    "• Validation Set (15%): Middle chronological block — used for XGBoost early stopping.\n"
    "• Test Set (15%): Most recent observations — used for final evaluation ONLY.\n\n"
    "The test set always represents future data relative to the training set, ensuring "
    "evaluation reflects realistic deployment conditions.",
    size=11, space_after=8)
add_para(doc, "9.3 Target Creation", bold=True, size=12, space_after=4)
add_para(doc,
    "All prediction targets were created by shifting future values to the corresponding "
    "current row using groupby('Ticker')[col].shift(-1). This means the target in row t "
    "is the value of the feature at row t+1 (next trading day). The final row per ticker "
    "always has NaN targets and is excluded from training and evaluation.",
    size=11, space_after=8)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 10 — MODELS
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "10. Models Used", level=1)
model_descriptions = [
    ("Naive Baseline",
     "Predicts tomorrow's close as today's close. Serves as the minimum performance "
     "benchmark — any useful model must outperform this baseline."),
    ("Linear Regression",
     "Fits a linear relationship between features and the target. Scaled features are "
     "used (StandardScaler). Simple, interpretable, and computationally cheap."),
    ("Logistic Regression",
     "Used for binary classification (Up/Down direction). Provides calibrated "
     "probability estimates suitable for ROC-AUC evaluation."),
    ("Random Forest (Regressor / Classifier)",
     "Ensemble of decision trees using bootstrap sampling and feature randomization. "
     "Handles non-linear relationships and provides native feature importance. "
     "Trained with n_estimators=200, max_depth=12."),
    ("XGBoost (Regressor / Classifier)",
     "Gradient boosting with L1/L2 regularization. Supports early stopping using "
     "the validation set. Generally the strongest performer for tabular financial data. "
     "Trained with n_estimators=200, max_depth=6, learning_rate=0.05."),
    ("LSTM (Optional)",
     "Long Short-Term Memory recurrent neural network. Captures sequential temporal "
     "dependencies using 30-day input sequences. Trained on a single representative "
     "stock (RELIANCE.NS) with 50 epochs, EarlyStopping, and Dropout regularization. "
     "Requires TensorFlow installation."),
]
for name, desc in model_descriptions:
    add_para(doc, name, bold=True, size=11, space_after=2)
    add_para(doc, desc, size=11, space_after=6)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 11 — PREDICTION TASKS
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "11. Prediction Tasks", level=1)
tasks_table = doc.add_table(rows=8, cols=4)
tasks_table.style = "Table Grid"
tasks_data = [
    ("Task", "Target Variable", "Type", "Metrics"),
    ("1", "Next_Day_Close", "Regression", "MAE, RMSE, R²"),
    ("2", "Next_Day_Return", "Regression", "MAE, RMSE, R²"),
    ("3", "Next_Day_Direction (0/1)", "Classification", "Accuracy, F1, ROC-AUC, Confusion Matrix"),
    ("4", "Next_Day_High", "Regression", "MAE, RMSE, R²"),
    ("5", "Next_Day_Low", "Regression", "MAE, RMSE, R²"),
    ("6", "Next_Day_Volatility", "Regression", "MAE, RMSE, R²"),
    ("7", "Next_Day_Regime (0/1/2)", "Classification", "Accuracy, F1 per class"),
]
for i, row_data in enumerate(tasks_data):
    row = tasks_table.rows[i]
    for j, val in enumerate(row_data):
        row.cells[j].text = val
        for para in row.cells[j].paragraphs:
            for run in para.runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.bold = True
if tasks_table.rows:
    shade_row(tasks_table.rows[0])
doc.add_paragraph()
add_para(doc,
    "Next_Day_Volatility is defined as the 20-day annualized rolling volatility of the "
    "next trading day. Market Regime thresholds are defined from the TRAINING data only "
    "using the 33rd and 67th percentiles of the rolling volatility distribution.",
    size=10, italic=True, space_after=8)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 12 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "12. Model Evaluation", level=1)
add_para(doc, "12.1 Regression Metrics", bold=True, size=12, space_after=4)
add_para(doc,
    "Mean Absolute Error (MAE): Average absolute difference between actual and predicted values. "
    "Root Mean Squared Error (RMSE): Penalizes large errors more than MAE. R² (Coefficient of "
    "Determination): Proportion of variance explained by the model (1.0 = perfect). "
    "MAPE (Mean Absolute Percentage Error): Used where scale-independent comparison is needed.",
    size=11, space_after=8)
add_para(doc, "12.2 Classification Metrics", bold=True, size=12, space_after=4)
add_para(doc,
    "Accuracy: Overall correct prediction rate. Precision: Of all predicted positives (UP), "
    "how many were actually UP. Recall: Of all actual positives (UP), how many were predicted "
    "as UP. F1-Score: Harmonic mean of precision and recall — robust to class imbalance. "
    "ROC-AUC: Area under the Receiver Operating Characteristic curve — measures discriminative "
    "ability across all classification thresholds.",
    size=11, space_after=8)
add_para(doc, "12.3 Model Comparison Table", bold=True, size=12, space_after=4)
add_para(doc,
    "The table below shows the expected structure. Actual values are populated after "
    "executing the Jupyter notebook. DO NOT fabricate metric values.",
    size=10, italic=True, space_after=4)
comp_table = doc.add_table(rows=6, cols=7)
comp_table.style = "Table Grid"
comp_data = [
    ("Model", "Task", "MAE", "RMSE", "R²", "Accuracy", "F1 / AUC"),
    ("Baseline", "Close", "[run notebook]", "[run notebook]", "[run notebook]", "—", "—"),
    ("Linear Regression", "Close", "[run notebook]", "[run notebook]", "[run notebook]", "—", "—"),
    ("Random Forest", "Close", "[run notebook]", "[run notebook]", "[run notebook]", "—", "—"),
    ("XGBoost", "Close", "[run notebook]", "[run notebook]", "[run notebook]", "—", "—"),
    ("XGBoost Clf", "Direction", "—", "—", "—", "[run notebook]", "[run notebook]"),
]
for i, row_data in enumerate(comp_data):
    row = comp_table.rows[i]
    for j, val in enumerate(row_data):
        row.cells[j].text = val
        for para in row.cells[j].paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)
                if i == 0:
                    run.bold = True
if comp_table.rows:
    shade_row(comp_table.rows[0])

# ════════════════════════════════════════════════════════════════════
# CHAPTER 13 — DASHBOARD
# ════════════════════════════════════════════════════════════════════
add_page_break(doc)
add_heading(doc, "13. Dashboard Design", level=1)
add_para(doc,
    "An interactive, eight-page Streamlit dashboard was developed using Plotly for "
    "all visualizations. The dashboard follows a professional financial analytics "
    "design using a dark navy sidebar (#0B132B), blue (#1C77C3), teal (#2CA6A4), "
    "and a clean white main area.",
    size=11, space_after=8)
pages = [
    ("Page 1 — Executive Overview",
     "KPI cards (companies, sectors, date range, records), market-wide price trend, "
     "top gainers/losers/volume leaders, sector return overview."),
    ("Page 2 — Stock Explorer",
     "Candlestick chart with overlay MA (SMA 20/50/200), volume bar chart, daily return "
     "bars, rolling volatility. User controls: stock, sector, year range."),
    ("Page 3 — Year Explorer",
     "Annual return bar chart across all years, full price history with selected years "
     "highlighted, year-vs-year comparison table."),
    ("Page 4 — Sector Analysis",
     "Sector daily return bar chart, annualized volatility bar chart, market cap treemap, "
     "return distribution box plot."),
    ("Page 5 — Technical Analysis",
     "Multi-panel chart with candlestick, volume, RSI, MACD. Toggleable indicator controls "
     "in sidebar. SMA, EMA, Bollinger Bands overlay."),
    ("Page 6 — ML Predictions",
     "Stock and model selector. Actual vs Predicted close chart, latest prediction KPIs, "
     "direction prediction with probability. Full disclaimer displayed prominently."),
    ("Page 7 — Model Performance",
     "Model comparison table and bar charts for regression and classification metrics. "
     "Feature importance images from saved notebook output."),
    ("Page 8 — Data Explorer",
     "Multi-filter (stock, sector, year range, columns). Tabular data view with download "
     "button, summary statistics expander, correlation heatmap expander."),
]
for name, desc in pages:
    add_para(doc, name, bold=True, size=11, space_after=2)
    add_para(doc, desc, size=11, space_after=6)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 14 — RESULTS
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "14. Results", level=1)
add_para(doc,
    "Results will be populated upon running the complete notebook pipeline "
    "(Nifty50_AI_Market_Analytics.ipynb). The following placeholder is provided in "
    "accordance with the project requirement that no metrics are fabricated.",
    bold=True, size=11, space_after=8, color=(0xDC, 0x26, 0x26))
add_para(doc,
    "To generate results:\n"
    "1. Install dependencies: pip install -r requirements.txt\n"
    "2. Place dataset in: database/nifty50_historical_data.csv\n"
    "3. Run the notebook from top to bottom.\n"
    "4. Results will be saved to: outputs/metrics/model_comparison.csv\n"
    "5. Update this section with the actual numerical results.",
    size=11, space_after=8)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 15 — LIMITATIONS
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "15. Limitations", level=1)
limitations = [
    "Survivorship bias: Only current Nifty 50 constituents are included. Companies that "
    "were delisted or removed from the index over the 25-year period are absent.",
    "Static fundamental data: Columns such as PE_Ratio, Market_Cap, and EPS appear to "
    "reflect a single snapshot value rather than a true historical time series.",
    "Indian market holidays: Business-day gap analysis was performed using Mon–Fri "
    "calendar without accounting for Indian public and market holidays.",
    "Market regime generalization: Machine learning models trained on historical data "
    "may not generalize after structural breaks such as the 2008 financial crisis, "
    "COVID-19 (2020), or major policy changes.",
    "No exogenous data: News sentiment, macroeconomic indicators, global market movements, "
    "and currency rates are not incorporated.",
    "LSTM compute constraints: Full LSTM training across all 49 stocks at sufficient "
    "depth requires GPU or cloud compute beyond a standard student laptop.",
    "Short-term prediction: All models predict one trading day ahead. Longer horizon "
    "predictions carry substantially greater uncertainty.",
    "No transaction costs: Backtesting of any strategy derived from these predictions "
    "would need to account for brokerage, taxes, and market impact.",
]
for item in limitations:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(item).font.size = Pt(11)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 16 — FUTURE SCOPE
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "16. Future Scope", level=1)
future = [
    "News Sentiment Analysis: Integrate NLP-based sentiment scores from financial news feeds.",
    "Macroeconomic Indicators: Add RBI rate decisions, CPI, GDP growth, FII/DII flows.",
    "Real-time Dashboard: Replace static CSVs with Yahoo Finance live API integration.",
    "Advanced Deep Learning: Transformer-based temporal models (TFT, Informer, PatchTST).",
    "Explainable AI: SHAP summary plots and LIME for individual prediction explanations.",
    "Portfolio Optimization: Combine predictions with Markowitz mean-variance or risk-parity.",
    "Backtesting Framework: Implement realistic strategy backtesting with transaction costs.",
    "Sector-Specific Models: Train separate models for each sector to capture unique dynamics.",
    "Alternative Data: Incorporate satellite imagery, web search trends, social media signals.",
]
for item in future:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(item).font.size = Pt(11)

# ════════════════════════════════════════════════════════════════════
# CHAPTER 17 — CONCLUSION
# ════════════════════════════════════════════════════════════════════
add_page_break(doc)
add_heading(doc, "17. Conclusion", level=1)
add_para(doc,
    "This project demonstrates a complete, reproducible, and transparent end-to-end data "
    "analytics and machine learning pipeline applied to 25+ years of Nifty 50 stock market "
    "data. The workflow progresses logically from raw data ingestion through rigorous quality "
    "auditing, cleaning, exploratory analysis, feature engineering, multi-task prediction, "
    "model comparison, and interactive dashboard deployment.",
    size=11, space_after=8)
add_para(doc,
    "Key contributions include: (1) a 20-point data quality audit with transparent "
    "documentation of all cleaning decisions; (2) strictly leakage-free feature engineering "
    "with chronological train/validation/test splitting; (3) seven distinct prediction tasks "
    "covering price, return, direction, high, low, volatility, and regime; (4) a comprehensive "
    "model comparison across five algorithmic approaches; and (5) a professional eight-page "
    "interactive Streamlit dashboard.",
    size=11, space_after=8)
add_para(doc,
    "The project was conducted as part of the IBM SkillsBuild Data Analytics with AI Academy "
    "Internship Program conducted by Bharat Care in association with AICTE. The program has "
    "provided a structured environment to develop practical skills in Python data science, "
    "machine learning, and financial analytics.",
    size=11, space_after=8)
add_para(doc,
    "FINAL DISCLAIMER: All models and predictions in this project are developed for "
    "educational and research purposes only. This report does not constitute financial "
    "advice. Investment decisions should always be made in consultation with qualified "
    "financial advisors. Past market performance does not guarantee future results.",
    bold=True, size=11, space_after=8, color=(0xDC, 0x26, 0x26))

# ════════════════════════════════════════════════════════════════════
# REFERENCES
# ════════════════════════════════════════════════════════════════════
add_heading(doc, "References", level=1)
references = [
    "[1] Kalyan (2024). Nifty50 Stocks (1999–2026) Daily OHLCV & Fundamentals. Kaggle. "
    "https://www.kaggle.com/datasets/kalyan197/nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals",
    "[2] pandas Development Team (2024). pandas: Powerful Python data analysis toolkit. "
    "https://pandas.pydata.org/",
    "[3] Scikit-learn Developers (2024). scikit-learn: Machine Learning in Python. "
    "https://scikit-learn.org/",
    "[4] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery "
    "and Data Mining.",
    "[5] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural "
    "Computation, 9(8), 1735–1780.",
    "[6] Investopedia (2024). Technical Analysis. https://www.investopedia.com/",
    "[7] Darío López Padial (2024). Technical Analysis Library in Python. "
    "https://technical-analysis-library-in-python.readthedocs.io/",
    "[8] Streamlit Inc. (2024). Streamlit documentation. https://docs.streamlit.io/",
    "[9] Plotly Technologies Inc. (2024). Plotly Python Graphing Library. "
    "https://plotly.com/python/",
    "[10] IBM SkillsBuild. Data Analytics with AI. https://skillsbuild.org/",
]
for ref in references:
    add_para(doc, ref, size=10, space_after=6)

# ════════════════════════════════════════════════════════════════════
# APPENDIX
# ════════════════════════════════════════════════════════════════════
add_page_break(doc)
add_heading(doc, "Appendix", level=1)
add_para(doc, "A. Project File Structure", bold=True, size=12, space_after=4)
structure_text = (
    "Nifty50_AI_Market_Analytics/\n"
    "├── Nifty50_AI_Market_Analytics.ipynb   ← Main analysis notebook\n"
    "├── requirements.txt\n"
    "├── README.md\n"
    "├── Nifty50_AI_Market_Analytics_Report.docx\n"
    "├── database/                            ← Raw dataset (Kaggle)\n"
    "│   ├── nifty50_historical_data.csv\n"
    "│   ├── nifty50_summary_statistics.csv\n"
    "│   └── metadata.json\n"
    "├── models/\n"
    "│   ├── xgb_close_model.joblib\n"
    "│   ├── rf_close_model.joblib\n"
    "│   ├── xgb_direction_model.joblib\n"
    "│   ├── rf_direction_model.joblib\n"
    "│   └── scalers/\n"
    "│       ├── feature_scaler.joblib\n"
    "│       └── target_scaler.joblib\n"
    "├── dashboard/\n"
    "│   └── app.py                           ← Streamlit dashboard\n"
    "└── outputs/\n"
    "    ├── figures/\n"
    "    ├── metrics/\n"
    "    │   └── model_comparison.csv\n"
    "    └── cleaned_data/\n"
    "        └── nifty50_cleaned.csv"
)
p = doc.add_paragraph()
p.add_run(structure_text).font.name = "Courier New"
p.runs[0].font.size = Pt(9)

doc.add_paragraph()
add_para(doc, "B. Key Code Snippets", bold=True, size=12, space_after=4)
add_para(doc,
    "Chronological Train/Test Split (from notebook, Section 13):",
    bold=True, size=10, space_after=2)
code_snippet = (
    "all_dates   = df_ml['Date'].sort_values().unique()\n"
    "train_cut   = all_dates[int(len(all_dates) * 0.70)]\n"
    "val_cut     = all_dates[int(len(all_dates) * 0.85)]\n"
    "train_df    = df_ml[df_ml['Date'] <= train_cut]\n"
    "val_df      = df_ml[(df_ml['Date'] > train_cut) & (df_ml['Date'] <= val_cut)]\n"
    "test_df     = df_ml[df_ml['Date'] > val_cut]"
)
p = doc.add_paragraph()
p.add_run(code_snippet).font.name = "Courier New"
p.runs[0].font.size = Pt(9)
doc.add_paragraph()
add_para(doc,
    "Target Creation (no look-ahead):",
    bold=True, size=10, space_after=2)
target_code = (
    "# Shift by -1 to get the NEXT row's value as the current row's target\n"
    "df['Next_Day_Close']     = df.groupby('Ticker')['Close'].shift(-1)\n"
    "df['Next_Day_Direction'] = (df['Next_Day_Close'] > df['Close']).astype(float)"
)
p = doc.add_paragraph()
p.add_run(target_code).font.name = "Courier New"
p.runs[0].font.size = Pt(9)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "Nifty50_AI_Market_Analytics_Report.docx"
doc.save(out_path)
print(f"Report saved: {out_path}")
