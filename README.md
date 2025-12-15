# Machine Learning Mini Projects

A collection of machine learning experiments exploring regression techniques and Retrieval-Augmented Generation (RAG) systems.

## Quick Start

```bash
# Clone and install dependencies
git clone <repository-url>
cd ml_mini_projects
pip install -r requirements.txt

# Launch Jupyter and explore
jupyter notebook
```

## What's Inside

### Regression Projects

**[Tips Prediction](Regression/tips/tips_regression.ipynb)** - Restaurant tipping analysis using Lasso and Linear Regression with GridSearchCV optimization.

**[Used Car Pricing](Regression/used_cars/used_car_price.ipynb)** - Price prediction on 426K Craigslist listings. Compares Linear, Lasso, Ridge, Polynomial, Random Forest, and XGBoost models. Features advanced encoding strategies and comprehensive EDA.

### 🤖 RAG (Retrieval-Augmented Generation)

**[PDF QA](RAG/qa_pdf_file/qa_with_pdf.ipynb)** - Question answering from PDF documents using ChromaDB and LangChain.

**[Text QA](RAG/qa_text_file/qa_with_text.ipynb)** - Basic RAG pipeline with State of the Union address.

**[Hybrid Search](RAG/qa_text_file_hybrid/qa_with_text_hybrid.ipynb)** - Combines semantic (vector) and keyword (BM25) search for robust retrieval

## Tech Stack

**ML:** scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn  
**RAG:** LangChain, ChromaDB, HuggingFace embeddings, OpenRouter API, rank-bm25

## Key Techniques Explored

- Regularization (Lasso, Ridge) and hyperparameter tuning
- Advanced feature encoding (target, ordinal, one-hot)
- Log transformations for skewed data
- Vector embeddings and semantic search
- Hybrid retrieval (BM25 + vector search)
- Document chunking and RAG pipelines

## Notes

- RAG projects require an [OpenRouter API key](https://openrouter.ai/keys)
- Used car dataset auto-downloads via `kagglehub`
- All notebooks include visualizations and evaluation metrics
