#!/usr/bin/env python3
"""
Experiment B4: The Definitive 10K FAISS Retrieval Ablation
Includes live progress bar (tqdm) and evaluates K up to 10.
"""
!pip install -q faiss-cpu sentence-transformers transformers datasets nltk torch tqdm
import json
import numpy as np
import random
from datasets import load_dataset
import nltk
from nltk import sent_tokenize
import warnings
import torch
from tqdm import tqdm  # Added for the progress bar!

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError as e:
    raise ImportError(f"Required package missing: {e}. Please run pip install first.")

try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

def embed_sentence(embedder, sentence: str) -> np.ndarray:
    emb = embedder.encode(sentence, convert_to_numpy=True)
    norm = np.linalg.norm(emb)
    if norm > 0: emb = emb / norm
    return emb.astype(np.float32)

def lexical_overlap(sent1: str, sent2: str) -> float:
    tokens1 = set(nltk.word_tokenize(sent1.lower()))
    tokens2 = set(nltk.word_tokenize(sent2.lower()))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0

def main():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_device = 0 if torch.cuda.is_available() else -1
    
    print(f"📥 Loading models (Using Device: {device_name.upper()})...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device_name) 
    
    print("📥 Loading RoBERTa (Batched for GPU)...")
    nli_model = pipeline("text-classification", model="roberta-large-mnli", 
                         top_k=None, device=pipeline_device, batch_size=16)

    print("📥 Loading halueval-llm-spans dataset...")
    ds = list(load_dataset("llm-semantic-router/halueval-llm-spans", split="train"))
    random.shuffle(ds)

    print("\n=======================================================")
    print(" 🔬 EXPERIMENT B4: FULL DATASET RECALL@K SWEEP")
    print("=======================================================\n")

    # --- CONFIGURATION ---
    MAX_SAMPLES_TO_TEST = 10000  # Set to total dataset size
    MAX_SAMPLES_TO_PRINT = 3     # Only print detailed logs for the first 3
    
    # ADDED 7 and 10 to the thresholds
    k_thresholds = [1, 2, 3, 5, 7, 10]
    recall_counts = {k: 0 for k in k_thresholds}
    mrr_sum = 0.0
    samples_tested = 0

    # Initialize the Progress Bar
    pbar = tqdm(total=MAX_SAMPLES_TO_TEST, desc="🔍 Testing FAISS Accuracy", unit="sample")

    for row in ds:
        spans = row.get('labels', [])
        if isinstance(spans, str):
            try: spans = json.loads(spans)
            except: continue

        conflict_span = next((s for s in spans if s.get('label', s.get('type', '')) == "Evident Conflict"), None)
        if not conflict_span: continue 

        answer_text = str(row.get('answer', ''))
        start = conflict_span.get('start', conflict_span.get('start_char', 0))
        end = conflict_span.get('end', conflict_span.get('end_char', 0))
        target_lie = answer_text[start:end].strip()

        if len(target_lie) < 10: continue 

        prompt_text = str(row.get('prompt', ''))
        if ":" in prompt_text[:100]:
            prompt_text = prompt_text.split(":", 1)[-1].strip()
        source_sentences = sent_tokenize(prompt_text)
        
        # We need at least 10 sentences to fairly test K=10
        if len(source_sentences) < 10: continue 
        
        best_truth_score = 0.0
        true_fact_idx = -1
        
        valid_sentences = [(idx, sent) for idx, sent in enumerate(source_sentences) if len(sent.strip()) > 5]
        if not valid_sentences: continue

        nli_inputs = [{"text": sent, "text_pair": target_lie} for idx, sent in valid_sentences]
        
        try:
            results = nli_model(nli_inputs)
            for (idx, sent), res in zip(valid_sentences, results):
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list): res = res[0]
                elif not isinstance(res, list): res = [res]
                
                for d in res:
                    lab = str(d.get("label", "")).lower()
                    if "contradiction" in lab or lab == "label_0":
                        score = float(d.get("score", 0.0))
                        if score > best_truth_score:
                            best_truth_score = score
                            true_fact_idx = idx
        except Exception: pass

        if best_truth_score < 0.75:
            continue
            
        samples_tested += 1
        should_print = (samples_tested <= MAX_SAMPLES_TO_PRINT)

        if should_print:
            # Using tqdm.write so it doesn't mess up the progress bar rendering
            tqdm.write(f"\n[{samples_tested}/{MAX_SAMPLES_TO_TEST}] =========================================")
            tqdm.write(f"🎯 TARGET LIE: \"{target_lie}\"")
            tqdm.write(f"✅ TRUE FACT (Score: {best_truth_score:.3f}): Sentence [{true_fact_idx}]")

        dim = embedder.get_sentence_embedding_dimension() if hasattr(embedder, 'get_sentence_embedding_dimension') else embedder.get_embedding_dimension()
        index = faiss.IndexFlatIP(dim)

        for sent in source_sentences:
            index.add(embed_sentence(embedder, sent).reshape(1, -1))
        
        lie_emb = embed_sentence(embedder, target_lie).reshape(1, -1)
        
        # SEARCH UP TO K=10 NOW
        D, I = index.search(lie_emb, min(10, len(source_sentences)))
        
        retrieved_indices = I[0].tolist()
        
        if should_print:
            tqdm.write("🔍 FAISS TOP 10 RETRIEVAL:")
            
        rank_found = -1
        for rank, source_idx in enumerate(retrieved_indices):
            if source_idx < 0 or source_idx >= len(source_sentences): continue
            
            sent_text = source_sentences[source_idx]
            overlap = lexical_overlap(target_lie, sent_text)
            
            is_match = (source_idx == true_fact_idx)
            if is_match and rank_found == -1:
                rank_found = rank + 1
                
            if should_print:
                status = "⭐ MATCH" if is_match else "❌ Miss "
                tqdm.write(f"   ➤ Rank {rank+1:<2} | Overlap: {overlap:.2f} | {status} | \"{sent_text[:60]}...\"")

        if rank_found > 0:
            mrr_sum += (1.0 / rank_found)
            for k in k_thresholds:
                if rank_found <= k:
                    recall_counts[k] += 1
                    
        # Update the progress bar
        pbar.update(1)

        if samples_tested >= MAX_SAMPLES_TO_TEST: 
            break

    pbar.close()

    print("\n=======================================================")
    print(f" 📊 FINAL THESIS METRICS (Over {samples_tested} Valid Samples)")
    print("=======================================================")
    print(f"Mean Reciprocal Rank (MRR): {mrr_sum/samples_tested:.3f} (1.0 is perfect)")
    print("\nCumulative Recall@K:")
    
    for k in k_thresholds:
        acc = (recall_counts[k] / samples_tested) * 100
        print(f"  ➤ Recall@{k:<2}: {acc:.1f}%")

if __name__ == "__main__":
    main()