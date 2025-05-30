import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import random
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables to avoid HuggingFace issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable for debugging CUDA errors

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class ScientificRankerDataset(Dataset):
    """Dataset for training the neural ranker with improved error handling"""
    
    def __init__(self, queries, candidates, labels, tokenizer, max_length=512):
        self.queries = queries
        self.candidates = candidates
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        assert len(queries) == len(candidates) == len(labels), "Input lengths must match"
        
        # Clean and validate data
        self.clean_data()
    
    def clean_data(self):
        """Clean and validate the dataset"""
        cleaned_queries = []
        cleaned_candidates = []
        cleaned_labels = []
        
        for i, (query, candidate, label) in enumerate(zip(self.queries, self.candidates, self.labels)):
            # Skip empty or None values
            if not query or not candidate or pd.isna(query) or pd.isna(candidate):
                logger.warning(f"Skipping empty data at index {i}")
                continue
            
            # Convert to string and clean
            query = str(query).strip()
            candidate = str(candidate).strip()
            
            # Skip if too short
            if len(query) < 3 or len(candidate) < 10:
                logger.warning(f"Skipping too short text at index {i}")
                continue
            
            # Ensure label is valid
            if label not in [0.0, 1.0]:
                label = float(label)
                if label < 0 or label > 1:
                    logger.warning(f"Invalid label {label} at index {i}, setting to 0")
                    label = 0.0
            
            cleaned_queries.append(query)
            cleaned_candidates.append(candidate)
            cleaned_labels.append(label)
        
        self.queries = cleaned_queries
        self.candidates = cleaned_candidates
        self.labels = cleaned_labels
        
        logger.info(f"Dataset cleaned: {len(self.queries)} samples remaining")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        try:
            query = self.queries[idx]
            candidate = self.candidates[idx]
            label = self.labels[idx]
            
            # Tokenize query and candidate together with error handling
            encoding = self.tokenizer(
                query,
                candidate,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(float(label), dtype=torch.float32)
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy sample to avoid breaking the dataloader
            dummy_text = "dummy text"
            encoding = self.tokenizer(
                dummy_text,
                dummy_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(0.0, dtype=torch.float32)
            }

class NeuralRanker(nn.Module):
    """Neural ranker model using BERT/SciBERT with improved stability"""
    
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.1):  # Changed default to more stable model
        super(NeuralRanker, self).__init__()
        
        # Try multiple strategies to load the model
        self.bert = None
        model_loaded = False
        
        # Strategy 1: Try with force_download=False and local_files_only for cached models
        try:
            self.bert = AutoModel.from_pretrained(
                model_name,
                return_dict=True,
                trust_remote_code=False,
                local_files_only=True  # Try cached version first
            )
            model_loaded = True
            logger.info(f"Loaded cached model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load cached {model_name}: {e}")
        
        # Strategy 2: Try downloading with compatibility settings
        if not model_loaded:
            try:
                self.bert = AutoModel.from_pretrained(
                    model_name,
                    return_dict=True,
                    trust_remote_code=False,
                    force_download=False,
                    resume_download=True
                )
                model_loaded = True
                logger.info(f"Downloaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to download {model_name}: {e}")
        
        # Strategy 3: Try alternative compatible models
        if not model_loaded:
            alternative_models = [
                'bert-base-uncased',
                'distilbert-base-uncased',
            ]
            
            for alt_model in alternative_models:
                if alt_model == model_name:  # Skip if already tried
                    continue
                try:
                    logger.info(f"Trying alternative model: {alt_model}")
                    # Try cached first
                    try:
                        self.bert = AutoModel.from_pretrained(
                            alt_model,
                            return_dict=True,
                            local_files_only=True
                        )
                    except:
                        # If no cache, download
                        self.bert = AutoModel.from_pretrained(
                            alt_model,
                            return_dict=True,
                            force_download=False
                        )
                    model_loaded = True
                    model_name = alt_model
                    logger.info(f"Successfully loaded {alt_model}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {alt_model}: {e}")
                    continue
        
        # Strategy 4: Try with minimal configuration for older PyTorch versions
        if not model_loaded:
            try:
                logger.info("Trying with basic configuration for older PyTorch...")
                from transformers import BertModel, BertConfig
                
                # Try to load a pre-trained config first
                try:
                    config = BertConfig.from_pretrained('bert-base-uncased')
                    self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)
                except:
                    # Fallback to default config
                    config = BertConfig(
                        vocab_size=30522,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072
                    )
                    self.bert = BertModel(config)
                    logger.warning("Using randomly initialized BERT model")
                
                model_loaded = True
                logger.info("Loaded with basic BERT configuration")
            except Exception as e:
                logger.error(f"Failed basic config: {e}")
        
        if not model_loaded:
            raise RuntimeError("Could not load any BERT model")
        
        # Get the hidden size from the model config
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 1)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        try:
            # Ensure input tensors are properly shaped and on correct device
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                # Fallback: use CLS token representation
                pooled_output = outputs.last_hidden_state[:, 0, :]
            
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            return logits.squeeze(-1)  # Remove last dimension if present
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return dummy output to prevent crash
            batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1
            return torch.zeros(batch_size, dtype=torch.float32, device=input_ids.device)

class ScientificClaimRanker:
    """Main class for training and inference with pure neural ranker"""
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer with compatibility handling
        self.tokenizer = self.load_tokenizer(model_name)
        self.model = NeuralRanker(model_name).to(self.device)
        self.tfidf = None
        self.collection_data = None
        self.collection_embeddings = None
        
    def load_tokenizer(self, model_name):
        """Load tokenizer with fallback options"""
        alternative_tokenizers = [model_name, 'bert-base-uncased', 'distilbert-base-uncased']
        
        for tokenizer_name in alternative_tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                # Ensure tokenizer has pad token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
                logger.info(f"Loaded tokenizer: {tokenizer_name}")
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
                continue
        
        raise RuntimeError("Could not load any tokenizer")
        
    def prepare_tfidf_retrieval(self, df_collection):
        """Prepare TF-IDF model for candidate retrieval"""
        logger.info("Preparing TF-IDF model...")
        
        # Handle missing values
        df_clean = df_collection.dropna(subset=['title', 'abstract'])
        logger.info(f"Cleaned collection: {len(df_clean)} papers (from {len(df_collection)})")
        
        corpus = df_clean[['title', 'abstract']].apply(
            lambda x: f"{str(x['title']).strip()} {str(x['abstract']).strip()}", axis=1
        ).tolist()
        
        # Filter out empty documents
        corpus = [doc for doc in corpus if len(doc.strip()) > 10]
        
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.collection_embeddings = self.tfidf.fit_transform(corpus)
        self.collection_data = df_clean
        logger.info("TF-IDF model prepared!")
    
    def get_tfidf_candidates(self, query, k=50):
        """Get top-k candidates using TF-IDF similarity"""
        if not query or len(query.strip()) < 3:
            logger.warning("Query too short for TF-IDF search")
            return []
            
        try:
            query_embedding = self.tfidf.transform([query.strip()])
            similarities = cosine_similarity(query_embedding, self.collection_embeddings).flatten()
            indices = np.argsort(-similarities)[:k]
            
            candidates = []
            for idx in indices:
                if idx < len(self.collection_data):
                    paper_data = self.collection_data.iloc[idx]
                    cord_uid = paper_data['cord_uid']
                    title = str(paper_data['title']).strip()
                    abstract = str(paper_data['abstract']).strip()
                    candidate_text = f"{title} {abstract}"
                    candidates.append((cord_uid, candidate_text, similarities[idx]))
            
            return candidates
        except Exception as e:
            logger.error(f"Error in TF-IDF candidate retrieval: {e}")
            return []
    
    def prepare_training_data_with_negatives(self, df_query, k_candidates=30, neg_samples=5):
        """Prepare training data with positive and negative samples"""
        logger.info("Preparing training data...")
        
        queries = []
        candidates = []
        labels = []
        
        # Clean query data
        df_query_clean = df_query.dropna(subset=['tweet_text', 'cord_uid'])
        logger.info(f"Processing {len(df_query_clean)} queries (from {len(df_query)})")
        
        for idx, row in tqdm(df_query_clean.iterrows(), total=len(df_query_clean)):
            try:
                query_text = str(row['tweet_text']).strip()
                true_cord_uid = str(row['cord_uid']).strip()
                
                if len(query_text) < 3:
                    continue
                
                # Get TF-IDF candidates
                tfidf_candidates = self.get_tfidf_candidates(query_text, k=k_candidates)
                
                if not tfidf_candidates:
                    continue
                
                # Find positive sample
                positive_found = False
                for cord_uid, candidate_text, score in tfidf_candidates:
                    if str(cord_uid).strip() == true_cord_uid:
                        queries.append(query_text)
                        candidates.append(candidate_text)
                        labels.append(1.0)
                        positive_found = True
                        break
                
                # If positive sample not in TF-IDF top-k, add it manually
                if not positive_found:
                    true_paper = self.collection_data[self.collection_data['cord_uid'] == true_cord_uid]
                    if not true_paper.empty:
                        true_paper_data = true_paper.iloc[0]
                        title = str(true_paper_data['title']).strip()
                        abstract = str(true_paper_data['abstract']).strip()
                        true_candidate_text = f"{title} {abstract}"
                        queries.append(query_text)
                        candidates.append(true_candidate_text)
                        labels.append(1.0)
                
                # Add negative samples from TF-IDF candidates
                neg_count = 0
                for cord_uid, candidate_text, score in tfidf_candidates:
                    if str(cord_uid).strip() != true_cord_uid and neg_count < neg_samples:
                        queries.append(query_text)
                        candidates.append(candidate_text)
                        labels.append(0.0)
                        neg_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing query {idx}: {e}")
                continue
        
        logger.info(f"Prepared {len(queries)} training samples")
        logger.info(f"Positive samples: {sum(labels)}")
        logger.info(f"Negative samples: {len(labels) - sum(labels)}")
        
        return queries, candidates, labels
    
    def train(self, df_query_train, df_query_dev, epochs=3, batch_size=8, learning_rate=2e-5, 
              k_candidates=30):
        """Train the neural ranker with improved error handling"""
        logger.info("Starting training...")
        
        try:
            # Prepare training data
            train_queries, train_candidates, train_labels = self.prepare_training_data_with_negatives(
                df_query_train, k_candidates=k_candidates, neg_samples=5
            )
            val_queries, val_candidates, val_labels = self.prepare_training_data_with_negatives(
                df_query_dev, k_candidates=k_candidates, neg_samples=3
            )
            
            if not train_queries or not val_queries:
                raise ValueError("No training or validation data prepared")
            
            # Create datasets
            train_dataset = ScientificRankerDataset(
                train_queries, train_candidates, train_labels, self.tokenizer, max_length=256
            )
            val_dataset = ScientificRankerDataset(
                val_queries, val_candidates, val_labels, self.tokenizer, max_length=256
            )
            
            # Create data loaders with error handling
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False,  # Reduce memory usage
                drop_last=True  # Avoid issues with batch size mismatch
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=True
            )
            
            # Setup optimizer and loss function
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            criterion = nn.BCEWithLogitsLoss()
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                train_pbar = tqdm(train_loader, desc="Training")
                for batch_idx, batch in enumerate(train_pbar):
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        # Ensure proper shapes
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0)
                        if attention_mask.dim() == 1:
                            attention_mask = attention_mask.unsqueeze(0)
                        if labels.dim() == 0:
                            labels = labels.unsqueeze(0)
                        
                        optimizer.zero_grad()
                        outputs = self.model(input_ids, attention_mask)
                        
                        # Ensure outputs and labels have same shape
                        if outputs.dim() != labels.dim():
                            if outputs.dim() > labels.dim():
                                outputs = outputs.squeeze()
                            else:
                                labels = labels.squeeze()
                        
                        loss = criterion(outputs, labels)
                        
                        # Check for NaN loss
                        if torch.isnan(loss):
                            logger.warning(f"NaN loss detected at batch {batch_idx}, skipping")
                            continue
                        
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        with torch.no_grad():
                            predictions = torch.sigmoid(outputs) > 0.5
                            train_correct += (predictions == (labels > 0.5)).sum().item()
                            train_total += labels.size(0)
                        
                        # Update progress bar
                        if batch_idx % 100 == 0:
                            train_pbar.set_postfix({
                                'loss': f"{loss.item():.4f}",
                                'acc': f"{train_correct/max(train_total, 1):.4f}"
                            })
                            
                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {e}")
                        continue
                
                if train_total == 0:
                    logger.error("No training samples processed successfully")
                    break
                
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = train_correct / train_total
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Validation")
                    for batch_idx, batch in enumerate(val_pbar):
                        try:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['label'].to(self.device)
                            
                            # Ensure proper shapes
                            if input_ids.dim() == 1:
                                input_ids = input_ids.unsqueeze(0)
                            if attention_mask.dim() == 1:
                                attention_mask = attention_mask.unsqueeze(0)
                            if labels.dim() == 0:
                                labels = labels.unsqueeze(0)
                            
                            outputs = self.model(input_ids, attention_mask)
                            
                            # Ensure outputs and labels have same shape
                            if outputs.dim() != labels.dim():
                                if outputs.dim() > labels.dim():
                                    outputs = outputs.squeeze()
                                else:
                                    labels = labels.squeeze()
                            
                            loss = criterion(outputs, labels)
                            
                            if not torch.isnan(loss):
                                val_loss += loss.item()
                                predictions = torch.sigmoid(outputs) > 0.5
                                val_correct += (predictions == (labels > 0.5)).sum().item()
                                val_total += labels.size(0)
                                
                        except Exception as e:
                            logger.error(f"Error in validation batch {batch_idx}: {e}")
                            continue
                
                if val_total == 0:
                    logger.error("No validation samples processed successfully")
                    break
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    try:
                        torch.save(self.model.state_dict(), 'best_neural_ranker_model.pth')
                        logger.info("Saved best model!")
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")
                        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def rank_with_tfidf_prefilter(self, query, prefilter_k=50, top_k=5):
        """Use TF-IDF to prefilter, then rank with neural model"""
        try:
            # Get TF-IDF candidates
            tfidf_candidates = self.get_tfidf_candidates(query, k=prefilter_k)
            
            if not tfidf_candidates:
                return []
            
            self.model.eval()
            scores = []
            
            with torch.no_grad():
                for cord_uid, candidate_text, tfidf_score in tfidf_candidates:
                    try:
                        encoding = self.tokenizer(
                            query,
                            candidate_text,
                            truncation=True,
                            padding='max_length',
                            max_length=256,
                            return_tensors='pt'
                        )
                        
                        input_ids = encoding['input_ids'].to(self.device)
                        attention_mask = encoding['attention_mask'].to(self.device)
                        
                        logit = self.model(input_ids, attention_mask)
                        score = torch.sigmoid(logit).item()
                        scores.append((cord_uid, score))
                        
                    except Exception as e:
                        logger.warning(f"Error ranking candidate {cord_uid}: {e}")
                        continue
            
            # Sort by neural score
            scores.sort(key=lambda x: x[1], reverse=True)
            return [cord_uid for cord_uid, _ in scores[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in ranking: {e}")
            return []
    
    def evaluate_ranker(self, df_query, prefilter_k=50):
        """Evaluate the ranker using MRR@5"""
        logger.info("Evaluating ranker...")
        
        predictions = []
        df_query_clean = df_query.dropna(subset=['tweet_text', 'cord_uid'])
        
        for idx, row in tqdm(df_query_clean.iterrows(), total=len(df_query_clean)):
            try:
                query_text = str(row['tweet_text']).strip()
                ranked_uids = self.rank_with_tfidf_prefilter(
                    query_text, prefilter_k=prefilter_k, top_k=5
                )
                predictions.append(ranked_uids)
            except Exception as e:
                logger.warning(f"Error evaluating query {idx}: {e}")
                predictions.append([])
        
        # Calculate MRR@5
        mrr_scores = []
        for i, (_, row) in enumerate(df_query_clean.iterrows()):
            true_uid = str(row['cord_uid']).strip()
            pred_uids = predictions[i]
            
            if true_uid in pred_uids:
                rank = pred_uids.index(true_uid) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        logger.info(f"MRR@5: {mrr:.4f}")
        
        return predictions, mrr
    
    def save_model(self, filepath='neural_ranker.pth'):
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown')
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath='neural_ranker.pth'):
        """Load a trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    """Main function using the full dataset for training and evaluation"""
    
    logger.info("Loading full datasets...")
    try:
        df_collection = pd.read_pickle('subtask4b_collection_data.pkl')
        df_query_train = pd.read_csv('subtask4b_query_tweets_train.tsv', sep='\t')
        df_query_dev = pd.read_csv('subtask4b_query_tweets_dev.tsv', sep='\t')
        logger.info(f"Loaded {len(df_collection)} papers, {len(df_query_train)} train queries, {len(df_query_dev)} dev queries")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize ranker
    logger.info("Initializing ranker...")
    try:
        ranker = ScientificClaimRanker(model_name='bert-base-uncased')
        logger.info("Ranker initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize ranker: {e}")
        return
    
    # Prepare TF-IDF retrieval
    logger.info("Preparing TF-IDF retrieval...")
    try:
        ranker.prepare_tfidf_retrieval(df_collection)
    except Exception as e:
        logger.error(f"Error preparing TF-IDF: {e}")
        return
    
    # Train using full datasets
    logger.info("Starting full training...")
    try:
        ranker.train(
            df_query_train,  # Full training set
            df_query_dev,    # Full dev set
            epochs=3,        # Increased epochs for full training
            batch_size=8,    # Original batch size
            learning_rate=2e-5,
            k_candidates=30
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    # Load best model
    try:
        ranker.model.load_state_dict(torch.load('best_neural_ranker_model.pth', 
                                               map_location=ranker.device))
        logger.info("Best model loaded!")
    except Exception as e:
        logger.warning(f"Could not load best model: {e}")
    
    # Evaluate on full dev set
    logger.info("Evaluating on full dev set...")
    try:
        dev_predictions, dev_mrr = ranker.evaluate_ranker(
            df_query_dev,  # Full dev set
            prefilter_k=50
        )
        
        # Save predictions
        df_submission = df_query_dev[['post_id']].copy()
        df_submission['preds'] = dev_predictions
        df_submission.to_csv('neural_ranker_predictions.tsv', sep='\t', index=False)
        
        logger.info(f"Final Dev MRR@5: {dev_mrr:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

if __name__ == "__main__":
    main()