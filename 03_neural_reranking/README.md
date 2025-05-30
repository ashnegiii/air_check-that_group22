
This repository contains code to train and evaluate a BERT-based neural re-ranking model
for retrieving the correct CORD-19 paper whenever a tweet mentions it implicitly.

Setup
-----
1. Create and activate a virtual environment:
   - Linux/macOS:
       python3 -m venv venv
       source venv/bin/activate
   - Windows (PowerShell):
       python3 -m venv venv
       .\\venv\\Scripts\\Activate.ps1

2. Install all required packages:
      pip install -r requirements.txt

Data Files
----------
Place the following files in the project root directory (do not rename):
  - subtask4b_collection_data.pkl
  - subtask4b_query_tweets_train.tsv
  - subtask4b_query_tweets_dev.tsv
  - subtask4b_query_tweets_test.tsv
  - subtask4b_query_tweets_test_gold.tsv   (for local evaluation only)

After training you will also have:
  - best_neural_ranker_model.pth   (430 MB checkpoint)

Training
--------
Run the training script:
    python neural_reranker.py

This will:
  1. Load train & dev tweet–paper pairs
  2. Build a TF-IDF candidate prefilter
  3. Sample positives & negatives
  4. Train the BERT cross-attention re-ranker for 3 epochs
  5. Save the best model to best_neural_ranker_model.pth

Note: Training may take several hours depending on hardware.

Evaluation & Submission
-----------------------
Once you have best_neural_ranker_model.pth, run:
    python evaluate.py

This will:
  1. Load the paper collection and test tweets
  2. Build the TF-IDF prefilter
  3. Load your .pth checkpoint (strict=False)
  4. Generate top-5 predictions for each tweet
  5. Compute local MRR@5 against test gold labels
  6. Write out predictions.tsv for Codalab

Note: Training can take hours and evaluation can take up to one to two hours.
