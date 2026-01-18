

## 📌 Project Overview

This project is an **interactive Streamlit-based dashboard** designed to analyze and visualize **commodity spot prices in Punjab (Pakistan)** with a special focus on **flood and post-flood impacts** on prices.

The dashboard integrates **historical data (2020–2024)**, **actual 2025 data**, and **model-based predictions** to show how extreme events (floods) influence commodity price trends.

It is suitable for:

* 📊 Data analysis & visualization
* 🌾 Agricultural economics
* 🌊 Climate & disaster impact studies
* 🎓 Academic projects (FYP, MS/BS)

---

## ✨ Key Features

* 📈 **Time-series visualization** of commodity spot prices (2020–2025)
* 📉 **4-week moving average smoothing** for trend clarity
* 🌊 **Flood & post-flood impact modeling** on prices
* 🔮 **Price prediction for future weeks**
* 🧮 **Linear Regression** for trend estimation
* 📊 **Interactive Plotly charts**
* 🧾 **Tabular flood impact summaries**
* 🎛️ **Commodity selection via dropdown**

---

## 🧪 Commodities Covered

* Potato (special case with seasonal modeling)
* Flour
* Milk
* Sugar
* Tomato
* Onion
* Chicken (Broiler)
* Pulses (Daal Chana, Masoor, Moong)
* Ghee & Cooking Oil variants
* Fertilizers (DAP)
* Bakery items (Naan)
* Citrus (Lemon)

*(Easily extendable by adding data columns)*

---

## 🗂️ Project Structure

```
project-folder/
│
├── 01.py or 09.py          # Main Streamlit application
├── amCharts (2).csv        # Potato spot price dataset
├── amchart,432.csv         # Other commodities dataset
├── README.md               # Project documentation
```

---

## 📊 Data Description

### 1️⃣ Potato Dataset (`amCharts (2).csv`)

* Weekly spot prices
* Years: 2020–2025
* Columns:

  * `Date`
  * `spot_price_2020` … `spot_price_2025`

### 2️⃣ Other Commodities Dataset (`amchart,432.csv`)

* Commodity-wise weekly prices
* Includes column:

  * `commodity_name`
  * `spot_price_2020` … `spot_price_2025`

⚠️ **Note:** 2026–2027 columns are removed intentionally to avoid data leakage.

---

## 🌊 Flood Impact Modeling

### Flood Weeks

```python
flood_weeks = [34, 35, 36, 37, 38, 39, 40, 41]
post_flood_weeks = [42, 43, 44, 45, 46]
```

### Impact Logic

* **Flood Weeks:** Price increases due to supply disruption
* **Post-Flood Weeks:** Residual effects at reduced intensity
* Impact factors are computed **from historical averages**, not hardcoded

---

## 🧠 Prediction Strategy

### Potato

* Seasonal averaging across 5 years
* Flood-adjusted price projection
* Continuous moving average (actual → predicted)

### Other Commodities

* Linear Regression on 2025 actual data
* Seasonal averaging (2020–2024)
* Blended prediction:

```
Predicted Price = (Trend + Seasonal Average) / 2
```

Flood multipliers applied conditionally.

---

## 📈 Visualization Details

* Plotly Line Charts
* Shaded flood & post-flood periods
* Dashed lines for predictions
* Vertical highlights for flood weeks
* Clean white theme for academic presentation

---

## 🛠️ Technologies Used

* **Python 3.9+**
* **Streamlit** – Web dashboard
* **Pandas & NumPy** – Data handling
* **Plotly** – Interactive charts
* **Scikit-learn** – Linear regression
* **GeoPandas** *(optional future extension)*

---

## ▶️ How to Run the App

### Step 1: Install Dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn geopandas
```

### Step 2: Navigate to Project Folder

```bash
cd "path/to/project-folder"
```

### Step 3: Run Streamlit App

```bash
streamlit run 01.py
```

(or)

```bash
streamlit run 09.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## ⚠️ Common Issues & Fixes

### ❌ FileNotFoundError

✔ Ensure CSV files are in the same folder as the script
✔ Use **relative paths**, not absolute Windows paths

### ❌ Streamlit warnings

✔ Always run using `streamlit run`, not `python file.py`

---

## 🎓 Academic Use (FYP / Viva)

This project demonstrates:

* Real-world data modeling
* Time-series analysis
* Disaster impact assessment
* Applied machine learning
* Interactive data storytelling

💡 Suitable for:

* Final Year Projects
* Research demos
* Policy simulations

---

## 🚀 Future Enhancements

* District-wise choropleth maps
* Export charts & tables
* LSTM-based forecasting
* User-uploaded datasets
* Real-time API integration

---

## 👤 Author

**Abdul Ahad**
Bioinformatics & Machine Learning Enthusiast
Punjab, Pakistan

---

