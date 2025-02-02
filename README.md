# TOPSIS-Based Evaluation of Pretrained Models for Text Similarity

This project evaluates various **pretrained NLP models** for **text similarity** using **cosine similarity** and ranks them using the **TOPSIS** method. The goal is to identify the best model for sentence similarity tasks.

## 📌 Features
- Uses **multiple transformer-based models** (BERT, RoBERTa, SBERT, GPT-2, DistilBERT).
- Computes **cosine similarity** between two sentences.
- Applies **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** to rank models.
- Outputs results in a **CSV file**.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/text-similarity-topsis.git
cd text-similarity-topsis
```

### 2️⃣ Install Dependencies
```sh
pip install torch transformers scipy numpy
```

---

## 🚀 Usage

Run the Python script:
```sh
python main.py
```

---

## 📊 How It Works

1️⃣ **Pretrained models** are loaded from Hugging Face.  
2️⃣ **Cosine similarity** is calculated for given sentence pairs.  
3️⃣ **TOPSIS** method ranks the models based on their similarity scores.  
4️⃣ **Results** are saved in `text_similarity_model_rankings.csv`.

---

## 📜 Example Sentences Used
```text
Sentence 1: "The rising inflation has impacted global markets significantly."
Sentence 2: "Global markets have been heavily affected due to increasing inflation rates."
```

---

## 📁 Output (CSV File)
| Model     | Rank |
|-----------|------|
| SBERT     | 1    |
| RoBERTa   | 2    |
| BERT      | 3    |
| DistilBERT | 4    |
| GPT-2     | 5    |

---

## 🖥 Models Used
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)
- **Sentence-BERT (SBERT)** (`sentence-transformers/bert-base-nli-mean-tokens`)
- **GPT-2** (`gpt2`)
- **DistilBERT** (`distilbert-base-uncased`)

---

## 📌 Dependencies
- `torch`
- `transformers`
- `scipy`
- `numpy`

Install all dependencies using:
```sh
pip install -r requirements.txt
```

---

## 🏆 Why Use This Project?
✅ Compare different **NLP models** for similarity tasks  
✅ Get a **ranked list** of models for better selection  
✅ **Fast and efficient** evaluation with pre-trained embeddings  

---

## 📜 License
This project is **open-source** under the MIT License.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

## 📞 Contact
For queries, open an **issue** or reach out via [your-email@example.com](mailto:your-email@example.com).  
Happy Coding! 🚀

