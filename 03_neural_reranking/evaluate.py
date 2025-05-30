import pandas as pd
import torch
from test import ScientificClaimRanker   

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2.1) Daten laden ---------------------------------------------------------
# (a) Collection-Index (Paper-Pool)
df_papers = pd.read_pickle("subtask4b_collection_data.pkl")

# (b) Test-Queries
df_test      = pd.read_csv("subtask4b_query_tweets_test.tsv",      sep="\t")
df_test_gold = pd.read_csv("subtask4b_query_tweets_test_gold.tsv", sep="\t")

# 2.2) Ranker initialisieren und Gewichte laden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ranker = ScientificClaimRanker(model_name="bert-base-uncased", device=device)

ranker.prepare_tfidf_retrieval(df_papers)

# Jetzt das .pth-File laden:
logger.info("Loading model weights...")
checkpoint = torch.load("best_neural_ranker_model.pth", map_location=device)
ranker.model.load_state_dict(checkpoint, strict=False)
ranker.model.to(device)
ranker.model.eval()

# 2.3)Top-5 Predictions erzeugen -------------------
preds = []
for tweet in df_test["tweet_text"]:
    top5 = ranker.rank_with_tfidf_prefilter(tweet, prefilter_k=50, top_k=5)
    preds.append(top5)

df_test["preds"] = preds


# Merge mit Gold
df_eval = df_test[["post_id", "preds"]].merge(
    df_test_gold[["post_id","cord_uid"]], on="post_id"
)

def mrr_at_5(golds, preds_list):
    total = 0.0
    for g, p in zip(golds, preds_list):
        try:
            rank = p.index(g) + 1
            total += 1.0 / rank
        except ValueError:
            pass
    return total / len(golds)

score = mrr_at_5(df_eval["cord_uid"], df_eval["preds"])
logger.info(f"Lokale Test-MRR@5: {score:.4f}")

# 2.5) predictions.tsv  -------------------------
df_submit = df_test[["post_id","preds"]].copy()
df_submit.to_csv("predictions.tsv", sep="\t", index=False)
logger.info("→ predictions.tsv ist fertig.")
