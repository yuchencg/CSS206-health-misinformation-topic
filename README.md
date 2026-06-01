# Dynamic Topic Modeling of Health Misinformation on Twitter (2015--2022)

Health-related misinformation filtered from labeled tweets; applied **LDA** and **BERTopic** to model how topics evolve across four two-year windows (2015--2022). 

## Data

The analysis uses the（[Truth Seeker dataset](https://www.unb.ca/cic/datasets/truthseeker-2023.html)）, which contains 134,198 tweets labeled as true or false. Three files are used:

| File | Description |
|------|-------------|
| `Truth_Seeker_Model_Dataset.csv` | Core dataset with tweet text and veracity labels |
| `Features_For_Traditional_ML_Techniques.csv` | 67 engineered NLP and user features |
| `Truth_Seeker_Model_Dataset_With_TimeStamps.xlsx` | Timestamps for temporal analysis |

Health-related tweets are filtered, yielding **33,254 health tweets** of which **73.8% are misinformation**.

## Methods

### Text Preprocessing

A 7-step pipeline: lowercasing, URL/mention removal, punctuation stripping, tokenization (NLTK), stopword removal (including domain-specific terms), lemmatization, and token filtering (alpha-only, 3+ characters).

### LDA

- K=6 topics, CountVectorizer with bigrams (max 3,000 features), batch learning
- Topic weights projected onto each time window to track prevalence shifts

### BERTopic

- Sentence-transformer embeddings + UMAP (5 components) + HDBSCAN (min cluster size 50)


### Comparison

| Time Window | Docs | LDA Dominant Topic | BERTopic Dominant Topic |
|-------------|------|--------------------|------------------------|
| 2017--2018 | 149 | vaccine, aluminum, brain, child | vaccine, aluminum, brain, child |
| 2019--2020 | 5,925 | covid, doctor, died, trump | vaccine, covid, child, death |
| 2021--2022 | 17,573 | vaccine, covid, mask, immunity | vaccine, covid, mask, vaccinated |

## Repository Structure

```
├── data/                          # Input datasets (not tracked in git)
├── output/                        # Generated visualizations and tables
│   ├── 01_fake_news_counts.png
│   ├── 02_lda_topic_weights.png
│   ├── 03_bertopic_over_time.png
│   ├── 04_bertopic_topics_over_time.png
│   ├── 04_bertopic_prevalence_heatmap.png
│   ├── 05_dominant_topic_wordclouds.png
│   └── 06_lda_vs_bertopic_comparison.csv
├── dynamic_topic_modeling_reproducible.ipynb
├── dynamic_topic_modeling_reproducible_latest_with_local_eval.ipynb
└── requirements.txt
```

### Notebooks

- **`dynamic_topic_modeling_reproducible.ipynb`** -- Main analysis pipeline: data loading, health filtering, preprocessing, LDA, BERTopic, and comparative visualizations
- **`dynamic_topic_modeling_reproducible_latest_with_local_eval.ipynb`** -- Extended version with additional evaluation metrics


## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn
- nltk, scikit-learn, scipy
- bertopic, umap-learn, hdbscan, sentence-transformers
- wordcloud, openpyxl


