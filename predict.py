import joblib
import string
import tkinter as tk
from tkinter import ttk, messagebox

category_map = {
    1: "World News",
    2: "Sports",
    3: "Business",
    4: "Science & Tech"
}

model = joblib.load("news_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def display_result(pred, confidence):
    label = category_map.get(pred, str(pred))
    result_text = f"Category: {label}\nConfidence: {round(confidence*100,2)}%"
    result_label.config(text=result_text)

def predict_text():
    input_text = text_box.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input missing", "Please enter some text to predict.")
        return
    cleaned = preprocess(input_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probas = model.predict_proba(vec)[0]
    confidence = max(probas)
    display_result(pred, confidence)

def clear_text():
    text_box.delete("1.0", tk.END)
    result_label.config(text="")

root = tk.Tk()
root.title("News Classifier Prediction")
root.geometry("650x420")
root.configure(bg="#212121")

#Styling ttk widgets
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TFrame", background="#212121")
style.configure("TLabel", background="#212121", foreground="#FAFAFA", font=("Segoe UI Semibold", 14))
style.configure("Header.TLabel", background="#212121", foreground="#3DDC97", font=("Segoe UI Bold", 22))
style.configure("TButton", background="#333", foreground="#FAFAFA", font=("Segoe UI Semibold", 12), borderwidth=0, padding=8)
style.map("TButton", background=[("active", "#2676FF"), ("pressed", "#2676FF")], foreground=[("active", "#FAFAFA"), ("pressed", "#FAFAFA")])



mainframe = ttk.Frame(root, padding=24, style="TFrame")
mainframe.pack(fill=tk.BOTH, expand=True)



ttk.Label(mainframe, text="News Classifier", style="Header.TLabel").pack(pady=(0,18))
ttk.Label(mainframe, text="Enter text to predict:", style="TLabel").pack(anchor="w")

text_box = tk.Text(mainframe, height=7, width=62, font=("Segoe UI", 13),
                   bg="#292929", fg="#FAFAFA", insertbackground="#3DDC97",
                   relief="flat", bd=2)
text_box.pack(pady=(2,10))

button_frame = ttk.Frame(mainframe, style="TFrame")
button_frame.pack(pady=7)

predict_btn = ttk.Button(button_frame, text="Predict", style="TButton", command=predict_text)
predict_btn.pack(side="left", padx=12)
clear_btn = ttk.Button(button_frame, text="Clear", style="TButton", command=clear_text)
clear_btn.pack(side="left", padx=12)

result_label = tk.Label(mainframe, text="", font=("Segoe UI", 15, "bold"),
                        fg="#3DDC97", bg="#212121", pady=18)
result_label.pack()

root.mainloop()
