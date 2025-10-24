# 📰 News Classifier GUI (Tkinter + ttk + Scikit‑Learn)

A **machine learning desktop app** built using **Tkinter’s ttk** widgets and **Scikit‑Learn**.  
This program classifies text (such as news articles) into categories using **TF‑IDF vectorization** and **Logistic Regression**.  
It includes a simple, modern GUI for making predictions with confidence scores — no command line needed!

***

## ⚙️ Key Features

✅ **TTK‑based modern UI**  
✅ **Prettified dark theme with custom colors**  
✅ **Category names mapped from numeric labels**  
✅ **TF‑IDF + Logistic Regression backend**  
✅ **Model files automatically saved with Joblib**  
✅ **Prediction GUI displays category + confidence**  
✅ **No Canvas required — pure ttk layout**

***

## 🧩 Technologies Used

| Component | Library |
|------------|----------|
| Interface | Tkinter + ttk |
| Data Processing | Pandas |
| NLP Preprocessing | NLTK |
| Vectorization | Scikit‑Learn TF‑IDF |
| Modeling | Logistic Regression |
| Model Saving | Joblib |

***

## 📁 Folder Structure

```
news-classifier/
│
├── train_gui.py                # (your own training script)
├── predict.py                  # the GUI app for text prediction
├── train.csv                   # training dataset
├── test.csv                    # testing dataset
├── news_classifier_model.pkl   # saved classifier
├── tfidf_vectorizer.pkl        # saved TF-IDF model
└── README.md                   # documentation (this file)
```

***

## 🚀 Quick Start

### Step 1 — Install Requirements  
```bash
pip install pandas scikit-learn nltk joblib
```

### Step 2 — Download NLTK Data  
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 3 — Train the Model  
Use your training script (example name: `train_gui.py`)  
to load CSV files and train the model with logistic regression.

This will output:
- `news_classifier_model.pkl`
- `tfidf_vectorizer.pkl`

***

## 🎯 Predict with the GUI

Once the model and vectorizer are ready,  
run the prediction GUI:

```bash
python predict.py
```

Then:
1. Type or paste any sample text in the text box.  
2. Press **Predict**.  
3. See your **category** and **confidence** displayed instantly below the buttons.  
4. Use **Clear** to reset inputs.

***

## 🧠 Example Category Mapping

In `predict.py`, the numeric classes are mapped to readable names:

```python
category_map = {
    1: "World News",
    2: "Sports",
    3: "Business",
    4: "Science & Tech"
}
```

Output example:
```
Category: Sports
Confidence: 94.3%
```

***

## 🎨 UI Color Palette

| Element | Color |
|----------|--------|
| Window Background | `#212121` |
| Text Box Background | `#292929` |
| Accent / Highlight | `#3DDC97` |
| Buttons (active) | `#2676FF` |
| Text Color | `#FAFAFA` |

***

## 💡 Dataset Format

Both `train.csv` and `test.csv` should include:

| Column | Description |
|--------|-------------|
| `Description` | The article body or text sample |
| `Class Index` | Class label (e.g. 0,1,2,3 ...) |

***

## 🪄 Example Use Case

| Input | Output |
|-------|---------|
| “The stock market saw record gains today.” | Category: Business |
| “NASA schedules new Mars rover mission.” | Category: Science & Tech |
| “The team won the championship finals.” | Category: Sports |

***

## 🧑‍💻 Author

Created with 💚 using **Tkinter’s ttk** and **Scikit‑Learn**.  
Focused on simplicity, readability, and a clean dark‑themed UI.

