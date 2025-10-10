import json
import os
import pandas as pd
import torch
import torch.nn as nn
import time
import sys
sys.setrecursionlimit(15000)
import math
from torch.optim import Adam
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve

class BaselineModel:
    """Baseline model that predicts follow-up probability based only on the overall ratio of follow-ups."""
    
    def __init__(self, df):
        self.df = df
        self.follow_up_ratio = None

    def train(self):
        """Train the baseline by computing the overall probability of a follow-up."""
        self.follow_up_ratio = self.df["follow_up"].mean()
        
        print("\n=== Follow-Up Probability ===")
        print(f"Overall Follow-Up Ratio: {self.follow_up_ratio:.4f}")

    def predict(self, num_samples):
        """Predict follow-up labels by sampling from Bernoulli distribution."""
        return np.random.choice([0, 1], size=num_samples, p=[1 - self.follow_up_ratio, self.follow_up_ratio]).astype(int)


class BaselineModel2:
    """Baseline model that predicts the probability of a follow-up based on historical conversation lengths."""
    
    def __init__(self, df):
        self.df = df
        self.probability_table = None

    def train(self):
        """Train the baseline by computing the probability of continuation and count for different conversation lengths."""
        # Group by 'turns' and calculate the mean and count of 'follow_up'
        group_stats = self.df.groupby("turns")["follow_up"].agg(['mean', 'size']).reset_index()
        group_stats.columns = ['turns', 'follow_up_probability', 'count']

        # Convert the DataFrame to a dictionary for easy access
        self.probability_table = group_stats.set_index('turns')['follow_up_probability'].to_dict()

        # Print the probability distribution with counts
        #print("\n=== Follow-Up Probability Distribution ===")
        #for _, row in group_stats.iterrows():
        #    print(f"Rounds: {row['turns']}, Count: {row['count']}, Follow-Up Probability: {row['follow_up_probability']:.4f}")


    def predict(self, turns):
        """Predict whether the conversation will continue based on probability."""
        probability = self.probability_table.get(turns, 0.5)  # Default to 50% if unseen turns
        return 1 if probability >= 0.5 else 0


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate precision, recall, F1-score, and optionally AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics = {"f1 micro": f1_micro, "f1 macro": f1}
    return metrics

def combine_user_requests(messages: List[Dict[str, Any]]) -> str:
    # (Keep implementation as before)
    user_msgs_content = []
    # Heuristic key detection
    role_key = "role" if messages and "role" in messages[0] else "from"
    content_key = "content" if messages and "content" in messages[0] else "value"
    user_tag = "user" if role_key == "role" else "human"

    for msg in reversed(messages):
        if len(user_msgs_content) >= 5: break
        if msg.get(role_key) == user_tag:
            user_msgs_content.append(msg.get(content_key, ""))
    user_msgs_content.reverse()
    n = len(user_msgs_content)
    total_word_limit = 512
    allowed_words_per_message = total_word_limit // n if n > 0 else 0
    processed_msgs = []
    if allowed_words_per_message > 0:
        for msg_content in user_msgs_content:
            words = msg_content.split()
            if len(words) > allowed_words_per_message:
                first_half_count = min(max(0, allowed_words_per_message // 2), len(words))
                second_half_count = min(max(0, allowed_words_per_message - first_half_count), len(words))
                if first_half_count + second_half_count > len(words):
                     processed_msg = " ".join(words[:allowed_words_per_message])
                else:
                     first_part = words[:first_half_count]
                     second_part = words[max(first_half_count, len(words)-second_half_count):]
                     processed_msg = " ".join(first_part + ["..."] + second_part)
            else:
                processed_msg = msg_content
            processed_msgs.append(processed_msg)
    elif user_msgs_content:
         processed_msgs.append(" ".join(user_msgs_content[0].split()[:total_word_limit]))
    combined_text = "\nThe next user query is: ".join(processed_msgs)
    return "query: " + combined_text


def load_conv_data(data_source, N, turn_equal_to, format_hint):
    print(f"Loading up to {N} samples from {format_hint} source...")
    conversation_features = []
    if isinstance(data_source, str) and data_source.endswith('.json'): 
        print(f"Loading from JSON file: {data_source}")
        f=open(data_source,'r',encoding='utf-8')
        data=json.load(f)
        f.close()
    elif isinstance(data_source, dict) and "train" in data_source:
        data=data_source["train"]
    else:
        data=data_source
    # train on last N conversations
    if N < 0:
        N = -N
        if isinstance(data, list):
            data = data[-N:]
        else:
            total_length = len(data)
            last_n_indices = list(range(total_length - N, total_length))
            data = data.select(last_n_indices)

    first_item=data[0] if len(data)>0 else {}
    role_tag, user_tag, value_tag, msg_list_key = 'role', 'user', 'content', 'conversation'
    if "conversations" in first_item: 
        role_tag, user_tag, value_tag, msg_list_key = 'from', 'human', 'value', 'conversations'
    elif "conversation_a" in first_item: 
        msg_list_key="conversation_a"
    print(f"Detected format keys: msg_list='{msg_list_key}', role='{role_tag}', content='{value_tag}'")
    processed_count=0
    for convo in data:
        if processed_count>=N:
            break
        messages = convo.get(msg_list_key,[])
        if not messages or not isinstance(messages, list) or len(messages)<2:
            continue
        turns=0
        for i in range(len(messages)-1):
            msg_i,msg_i_plus_1 = messages[i],messages[i+1]
            assistant_tags = ("assistant","gpt")
            if (msg_i.get(role_tag)==user_tag and msg_i_plus_1.get(role_tag) in assistant_tags):
                if turns <= turn_equal_to:
                    has_follow_up = (i+2<len(messages) and messages[i+2].get(role_tag)==user_tag)
                    true_label=1 if has_follow_up else 0
                    combined_text_input = combine_user_requests(messages[:i+1])
                    conversation_features.append({
                        "follow_up":true_label, "turns":turns, 'text':combined_text_input
                    })
                turns+=1
                processed_count+=1
                if turns > turn_equal_to:
                    break

    print(f"Finished processing. Extracted {len(conversation_features)} features.")
    if not conversation_features: raise ValueError("No features extracted.")
    return pd.DataFrame(conversation_features)


# --- Dataset Class --- (Needed by prepare_dataloaders)
class ConversationTurnDataset(Dataset):
# (Implementation remains unchanged)
    def __init__(self, df: pd.DataFrame, task: str = "classification"):
        self.df = df; self.task = task
        self.label_name = "follow_up" if task == "classification" else "tta"
        self.label_dtype = torch.long if task == "classification" else torch.float
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; text = str(row.get("text", ""))
        turns = row.get("turns", 0)
        label_val = row.get(self.label_name, 0 if self.task == "classification" else 0.0)
        return text, turns, torch.tensor(label_val, dtype=self.label_dtype)

# --- Standalone Data Preparation Function (Modified) ---
def prepare_dataloaders(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    batch_size: int = 16,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, pd.DataFrame]: # Now returns test_df as well
    """
    Splits DataFrame, creates train/test DataLoaders, returns test DataFrame.
    """
    label_col = "follow_up" if task == "classification" else "tta"
    if label_col not in df.columns: raise ValueError(f"Label column '{label_col}' not found.")
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, 
            stratify=df[label_col] if task=='classification' and df[label_col].nunique() > 1 else None )
    except Exception as e:
        print(f"Could not stratify split (Error: {e}). Performing regular split.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Data split: {len(train_df)} train, {len(test_df)} test samples.")
    train_dataset = ConversationTurnDataset(train_df, task=task)
    test_dataset = ConversationTurnDataset(test_df, task=task)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # Test loader not strictly needed by train if evaluating row-by-row, but can be useful elsewhere
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print(f"Created DataLoaders with batch size {batch_size}.")
    # Return the test DataFrame for row-by-row evaluation
    return train_dataloader, test_dataloader, train_df, test_df


# --- MLP Classifier Definition ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, num_layers=3, dropout=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        current_dim = input_dim
        if num_layers == 1:
             layers.append(nn.Linear(current_dim, output_dim))
        else:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


# --- Main ML Model Class (Refactored) ---
class MLModel:
    """ ML-based model using Sentence Transformers and MLP classifier. """
    def __init__(self, bert_model_name: str = "intfloat/multilingual-e5-small", 
                 #"sentence-transformers/all-MiniLM-L6-v2", 
                 hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.3, task: str = "classification", 
                 dataset_choice: Optional[str] = None, device: Optional[str] = None ):
        # (Initialization remains the same - no data attributes)
        self.text_to_embedding = {}
        self.bert_model_name = bert_model_name; self.hidden_dim = hidden_dim
        self.num_layers = num_layers; self.dropout = dropout
        self.task = task; self.dataset_choice = dataset_choice
        self.best_threshold = 0.5
        if device: 
            self._device = torch.device(device)
        else: 
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 384 + 1
        self.output_dim = 1 if self.task == "regression" else 2
        self.classifier = MLPClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, 
                                        output_dim=self.output_dim, num_layers=self.num_layers, 
                                        dropout=self.dropout ).to(self._device)
        self.y_true = []
        self.y_pred = []

    def train(
        self,
        train_dataloader: DataLoader,
        train_df,
        test_df: pd.DataFrame, # Requires the test DataFrame for evaluation
        num_epochs: int = 6,
        lr: float = 2e-5,
        no_embedding: bool = False,
        save_dir: str = "checkpoints"
        ):
        """ Train the model and evaluate using single predictions on test_df after each epoch. """
        self.classifier.train()
        
        if self.task == "classification":
            counts = train_df['follow_up'].value_counts().sort_index().values  # [neg_count, pos_count]
            total = counts.sum()
            class_weights = [ total/counts[0], total/counts[1] ]
            weights = torch.tensor(class_weights, device=self._device, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.MSELoss()
        optimizer = Adam(self.classifier.parameters(), lr=lr)
        best_test_metric = float('inf') # Use -inf since lower loss or higher F1 is better
        label_col = "follow_up" if self.task == "classification" else "tta"

        print(f"\n--- Starting Training ---")
        for epoch in range(num_epochs):
            self.classifier.train() # Ensure train mode for training phase
            total_train_loss = 0
            start_time_epoch = time.time()

            # --- Training Phase ---
            for batch_idx, batch in enumerate(train_dataloader):
                try: 
                    input_texts, input_vals, labels = batch
                except ValueError:
                    print(f"Skipping malformed batch {batch_idx}"); continue
                labels = labels.to(self._device)
                input_vals = input_vals.to(self._device).float().unsqueeze(1)
                if self.task == 'regression': 
                    labels = labels.float().unsqueeze(1)
                else: 
                    labels = labels.long()
                with torch.no_grad():
                    if epoch > 0:
                        embeddings = []
                        for text in list(input_texts):
                            embedding = self.text_to_embedding[text]
                            embeddings.append(embedding)
                        embeddings = torch.stack(embeddings)
                    else:
                        embeddings = self.bert.encode(list(input_texts), convert_to_tensor=True, 
                                                    batch_size=len(input_texts))
                        for i in range(len(embeddings)):
                            self.text_to_embedding[input_texts[i]] = embeddings[i]
                if no_embedding: 
                    embeddings.zero_()
                combined_features = torch.cat((embeddings, input_vals), dim=1)
                logits = self.classifier(combined_features)
                loss = criterion(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); total_train_loss += loss.item()
                # (Removed intra-epoch progress printing for brevity, can be added back)

            avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
            print(f"--- Epoch {epoch+1} Training Summary ---")
            print(f"Time: {(time.time() - start_time_epoch):.2f}s, Avg Train Loss: {avg_train_loss:.4f}")

            # --- Evaluation Phase (Instance-by-Instance) ---
            print(f"--- Starting Epoch {epoch+1} Evaluation (using predict_single_processed) ---")
            self.classifier.eval() # Set model to evaluation mode
            y_true_eval = []
            y_pred_eval = []
            y_prob_eval = [] # Store probabilities for potential AUC calculation
            test_loss_sum = 0
            start_time_eval = time.time()

            for idx, row in test_df.iterrows():
                true_label = row[label_col]
                preprocessed_text = str(row['text']) # Ensure text is string
                turns = int(row['turns'])

                # Get prediction/probability using the single prediction method
                if self.task == "classification":
                    logit = self.predict_single_processed(preprocessed_text, turns,
                                                         no_embedding=no_embedding, return_prob=False)
                    loss = criterion(logit, torch.tensor(true_label, dtype=torch.long).unsqueeze(0).to(self._device))
                    test_loss_sum += loss.item()
                    prob = torch.softmax(logit, dim=1)[0, 1].item()
                    pred_label = 1 if prob > 0.5 else 0 # Simple thresholding
                    y_prob_eval.append(prob)
                    y_pred_eval.append(pred_label)
                else: # Regression
                    pred_value = self.predict_single_processed(preprocessed_text, turns, 
                                                               no_embedding=no_embedding)
                    y_prob_eval.append(pred_value)
                    y_pred_eval.append(pred_value)

                y_true_eval.append(true_label)
            # =======
            precision, recall, thresholds = precision_recall_curve(y_true_eval, y_prob_eval)
            f1_scores = 2*precision*recall/(precision+recall+1e-12)
            best_idx = f1_scores.argmax()
            self.best_threshold = thresholds[best_idx]
            print(f"Use threshold {self.best_threshold:.3f} for max F1 of {f1_scores[best_idx]:.3f}")
            # =======

            eval_time = time.time() - start_time_eval
            print(f"--- Epoch {epoch+1} Evaluation Summary ---")
            print(f"Time: {eval_time:.2f}s")

            # Calculate final metrics for the epoch
            final_metrics = {}
            current_metric_val = -float('inf')
            if self.task == "classification" and len(y_true_eval) > 0:
                final_metrics = calculate_metrics(y_true_eval, y_pred_eval, y_prob_eval)
                print(f"Precision={final_metrics['precision']:.4f}, \
                      Recall={final_metrics['recall']:.4f}, F1={final_metrics['f1']}")
                # current_metric_val = (final_metrics['f1'][0] + final_metrics['f1'][1]) / 2
                current_metric_val = test_loss_sum / len(y_prob_eval)
                print("test loss: ", current_metric_val)
                mcc = matthews_corrcoef(y_true_eval, y_pred_eval)
                current_metric_val = -mcc
                print(f"MCC: {mcc:.4f}")
            elif self.task == "regression" and len(y_true_eval) > 0:
                # Calculate final regression metric (e.g., MSE)
                final_mse = mean_squared_error(y_true_eval, y_pred_eval)
                final_metrics = {'mse': final_mse}
                print(f"Final Eval MSE: {final_mse:.4f}")
                current_metric_val = -final_mse # Use negative MSE (higher is better)

            # Save the best model
            if current_metric_val <= best_test_metric:
                best_test_metric = current_metric_val
                metric_str = f"{current_metric_val:.4f}".replace('.', '_').replace('-', '')
                save_filename = f"{self.dataset_choice or 'model'}_metric_{metric_str}_epoch_{epoch}.pt"
                save_path = os.path.join(save_dir, save_filename)
                self.save_model(save_path)
                print(f"*** New best model saved to {save_path} (Eval Metric: {current_metric_val:.4f}) ***")

            # Crucial: Set back to train mode for the next epoch's training phase
            self.classifier.train()

        print("--- Training Finished ---")
        # Ensure model is in eval mode after all epochs are done
        self.classifier.eval()


    def predict_batch_processed(
        self,
        preprocessed_texts: list,  # List of preprocessed texts
        turns: list,
        no_embedding: bool = False,
        return_prob: bool = True
    ) -> list:
        """
        Predict outcomes for a batch of instances using preprocessed texts.
        Returns a list of probabilities or predictions.
        """
        self.classifier.eval()  # Ensure evaluation mode
        input_vals = torch.tensor(turns, dtype=torch.float).unsqueeze(1).to(self._device)
        with torch.no_grad():
            embeddings = self.bert.encode(preprocessed_texts, convert_to_tensor=True, batch_size=len(preprocessed_texts))
            if no_embedding:
                embeddings.zero_()
            combined_features = torch.cat((embeddings, input_vals), dim=1)
            logits = self.classifier(combined_features)
        if self.task == "regression":
            return logits.squeeze().cpu().numpy().tolist()
        else:  # Classification
            probabilities = torch.softmax(logits, dim=1)
            if return_prob:
                return probabilities[:, 1].cpu().numpy().tolist()  # Prob of class 1 for each sample
            else:
                return torch.argmax(probabilities, dim=1).cpu().numpy().tolist()

    def predict_single_processed(
        self,
        preprocessed_text: str, # Takes preprocessed text now
        turns: int,
        true_label = -1,
        no_embedding: bool = False,
        return_prob: bool = True
    ) -> Union[float, int]:
        """
        Predicts outcome for a single instance using PREPROCESSED text.
        Performs direct inference.
        """
        self.classifier.eval() # Ensure evaluation mode
        
        # print(preprocessed_text, turns)

        # 1. Prepare input tensors
        input_vals = torch.tensor([[turns]], dtype=torch.float).to(self._device)

        # 2. Encode text (already preprocessed)
        with torch.no_grad():
            # Note: Encoding even a single sentence has overhead.
            embeddings = self.bert.encode([preprocessed_text], convert_to_tensor=True, batch_size=1)
            if no_embedding:
                embeddings.zero_()

            # 3. Combine features
            combined_features = torch.cat((embeddings, input_vals), dim=1)

            # 4. Perform inference
            logits = self.classifier(combined_features)

        # 5. Process output
        if self.task == "regression":
            return logits.item()
        else: # Classification
            probabilities = torch.softmax(logits, dim=1)
            pred_label = 1 if probabilities[0, 1].item() > self.best_threshold else 0
            pred_label = torch.argmax(probabilities, dim=1).item() # Label 0 or 1
            if true_label >= 0:
                self.y_true.append(true_label)
                self.y_pred.append(pred_label)
                if len(self.y_true) % 100 == 0:
                    metrics = calculate_metrics(self.y_true, self.y_pred)
                    print(f"Precision={metrics['precision']:.4f}, \
                      Recall={metrics['recall']:.4f}, F1={metrics['f1']}")
                    mcc = matthews_corrcoef(self.y_true, self.y_pred)
                    print(f"MCC: {mcc:.4f}")
            if return_prob:
                return probabilities[0, 1].item() # Prob of class 1
            else:
                return logits

    def save_model(self, save_path: str):
        save_content = {'model_state_dict': self.classifier.state_dict(), 
                        'config': {'bert_model_name': self.bert_model_name, 
                                   'hidden_dim': self.hidden_dim, 
                                   'num_layers': self.num_layers, 
                                   'dropout': self.dropout, 
                                   'task': self.task, 
                                   'input_dim': self.input_dim, 
                                   'output_dim': self.output_dim,
                                   'best_threshold': self.best_threshold
                                   } 
                        }
        torch.save(save_content, save_path)
        print(f"Saved to {save_path}")

    def load_model(self, model_path: str):
        if not os.path.exists(model_path): 
            raise FileNotFoundError(f"Model file not found at {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        if 'config' in checkpoint:
            config = checkpoint['config']; print("Loading model configuration from checkpoint...")
            self.bert_model_name = config.get('bert_model_name', self.bert_model_name)
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            self.num_layers = config.get('num_layers', self.num_layers)
            self.dropout = config.get('dropout', self.dropout)
            self.task = config.get('task', self.task)
            self.best_threshold = config.get('best_threshold', self.best_threshold)
            input_dim = config.get('input_dim', self.input_dim)
            output_dim = config.get('output_dim', self.output_dim)
            if self.bert_model_name == "intfloat/multilingual-e5-small":
                my_path = "/scratch/gpfs/dy5/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/c007d7ef6fd86656326059b28395a7a03a7c5846"
                if os.path.exists(my_path):
                    self.bert_model_name = my_path
            print(f"Re-initializing BERT with: {self.bert_model_name}")
            self.bert = SentenceTransformer(self.bert_model_name, device=str(self._device))
            self.bert.eval()
            print(f"Re-initializing Classifier with HParams: \
                  hidden={self.hidden_dim}, layers={self.num_layers}, dropout={self.dropout}")
            self.classifier = MLPClassifier(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim, 
                                            num_layers=self.num_layers, dropout=self.dropout).to(self._device)
            self.classifier = torch.jit.script(self.classifier)
        else: print("Warning: No config found in checkpoint.")
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else: 
            self.classifier.load_state_dict(checkpoint) # Legacy
        self.classifier.to(self._device); self.classifier.eval()
        print(f"Classifier state_dict loaded from {model_path} and set to eval mode.")


# ==== RUN MODEL ====
if __name__ == "__main__":
    task_type = "classification"
    dataset_choice = 'sharegpt'
    #dataset_choice = 'chatbot_arena' #'tay' 'gpt4' #"chatbot_arena" #"sharegpt" # "lmsys-chat-1m"
    N = 20000

    turn_equal_to = 20
    if dataset_choice == "lmsys-chat-1m":
        print("Loading dataset: LMSys-chat-1M from Hugging Face...")
        ds = load_dataset("lmsys/lmsys-chat-1m")  # Use authentication if required
        df = load_conv_data(ds, N, turn_equal_to, 'lmsys')
    elif dataset_choice == "sharegpt":
        print(f"Loading dataset: ShareGPT")
        df = load_conv_data("ShareGPT_V3_unfiltered_cleaned_split.json", N, turn_equal_to, 'sharegpt')
    elif dataset_choice == "tay":
        print(f"Loading dataset: Tay")
        df = load_conv_data("../../tay.json", N, turn_equal_to, 'tay')
    elif dataset_choice == "chatbot_arena":
        print("Loading dataset: Chatbot Arena Conversations from Hugging Face...")
        ds = load_dataset("lmsys/chatbot_arena_conversations")
        df = load_conv_data(ds, N, turn_equal_to, 'chatbot_arena')
    elif dataset_choice == "gpt4":
        print("Loading dataset: GPT4 Conversations from Hugging Face...")
        ds = load_dataset("lightblue/gpt4_conversations_multilingual")
        df = load_conv_data(ds, N, turn_equal_to, 'gpt4')
    else:
        print('dataset not found')
    train_loader, test_loader, train_df, test_df = prepare_dataloaders( # Get test_df back
        df=df, task=task_type, batch_size=256, test_size=0.5
    )

    baseline = BaselineModel(train_df)
    baseline.train()
    y_true = test_df['follow_up'].values
    y_pred = baseline.predict(len(test_df))
    mcc_baseline = matthews_corrcoef(y_true, y_pred)
    print(f"BaselineModel MCC: {mcc_baseline:.4f}")
    metrics = calculate_metrics(y_true, y_pred)
    print(f"BaselineModel F1 micro: {metrics['f1 micro']:.4f}, F1 macro: {metrics['f1 macro']:.4f}")
    baseline2 = BaselineModel2(train_df)
    baseline2.train()
    y_pred2 = test_df['turns'].apply(baseline2.predict).values
    mcc_baseline2 = matthews_corrcoef(y_true, y_pred2)
    print(f"BaselineModel2 MCC: {mcc_baseline2:.4f}")
    metrics2 = calculate_metrics(y_true, y_pred2)
    print(f"BaselineModel2 F1 micro: {metrics2['f1 micro']:.4f}, F1 macro: {metrics2['f1 macro']:.4f}")

    # --- Initialize Model ---
    mlmodel = MLModel(task=task_type, dataset_choice=dataset_choice)

    # --- Train Model (passing train_loader and test_df) ---
    checkpoint_dir = f"../vllm/benchmarks/checkpoints_{dataset_choice}_{turn_equal_to}"
    if N < 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        mlmodel.train(
            train_dataloader=train_loader,
            train_df=train_df,
            test_df=test_df, # Pass test_df for evaluation within train
            num_epochs=20,
            lr=5e-5,
            save_dir=checkpoint_dir
        )
    # --- Example: Load Best Model & Use Prediction Methods ---
    try:
        saved_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        saved_files.sort(reverse=False)
        best_checkpoint_path = os.path.join(checkpoint_dir, saved_files[0])
        print(f"\n--- Loading Best Checkpoint: {best_checkpoint_path} ---")
        # Re-initialize or use existing model instance and load weights
        # We need to ensure HParams match if re-initializing without config in checkpoint
        mlmodel_loaded = MLModel(task=task_type, dataset_choice=dataset_choice)
        mlmodel_loaded.load_model(best_checkpoint_path) # load_model sets to eval()

        # --- test ---
        sum_true_label = 0
        start_time = time.time()
        total_preds = 0
        all_true = []
        all_probs = []
        for batch in train_loader:
            batch_texts, batch_turns, batch_labels = batch
            batch_texts = list(batch_texts)
            batch_turns = batch_turns.tolist()
            batch_labels = batch_labels.tolist()
            sum_true_label += sum(batch_labels)
            batch_probs = mlmodel_loaded.predict_batch_processed(batch_texts, batch_turns)
            total_preds += len(batch_probs)
            all_true.extend(batch_labels)
            all_probs.extend(batch_probs)
        #print("has follow up: ", sum_true_label, "/", total_preds)
        #print("per prediction time: ", (time.time() - start_time) / total_preds)
        # Calculate best threshold and print metrics
        precision_curve, recall_curve, thresholds = precision_recall_curve(all_true, all_probs)
        # Best threshold by max MCC
        mccs = []
        for t in thresholds:
            y_pred_t = [1 if prob > t else 0 for prob in all_probs]
            mccs.append(matthews_corrcoef(all_true, y_pred_t))
        mccs = np.array(mccs)
        best_mcc_idx = mccs.argmax()
        best_mcc_threshold = thresholds[best_mcc_idx] if best_mcc_idx < len(thresholds) else 0.5
        max_mcc = mccs[best_mcc_idx]
        print(f"Best threshold (MCC): {best_mcc_threshold:.4f} for max MCC: {max_mcc:.4f}")
        
        total_preds = 0
        all_true = []
        all_probs = []
        for batch in test_loader:
            batch_texts, batch_turns, batch_labels = batch
            batch_texts = list(batch_texts)
            batch_turns = batch_turns.tolist()
            batch_labels = batch_labels.tolist()
            sum_true_label += sum(batch_labels)
            batch_probs = mlmodel_loaded.predict_batch_processed(batch_texts, batch_turns)
            total_preds += len(batch_probs)
            all_true.extend(batch_labels)
            all_probs.extend(batch_probs)
        y_pred_mcc = [1 if prob > best_mcc_threshold else 0 for prob in all_probs]
        max_mcc = matthews_corrcoef(all_true, y_pred_mcc)
        precision_mcc = precision_score(all_true, y_pred_mcc, zero_division=0)
        recall_mcc = recall_score(all_true, y_pred_mcc, zero_division=0)
        f1_mcc = f1_score(all_true, y_pred_mcc, average='micro', zero_division=0)
        f1_macro_mcc = f1_score(all_true, y_pred_mcc, average='macro', zero_division=0)
        f1_per_class_mcc = f1_score(all_true, y_pred_mcc, average=None, zero_division=0)
        print(f"f1 micro={f1_mcc:.4f}, f1 macro={f1_macro_mcc:.4f}, MCC={max_mcc:.4f}")
    except IndexError: print(f"\nNo checkpoints found in '{checkpoint_dir}'.")
    except Exception as e: print(f"\nError during loading/prediction example: {e}")
