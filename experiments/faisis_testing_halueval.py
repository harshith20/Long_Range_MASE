#!/usr/bin/env python3
"""
Experiment A: FAISS Retrieval Accuracy Test
Isolates the FAISS index and tests its ability to retrieve contradictory
evidence for known "Evident Conflict" hallucinations.
"""

# Ensure all required packages are installed
!pip install -q faiss-cpu sentence-transformers transformers datasets

import sys
import json
import numpy as np
from datasets import load_dataset
import nltk
from nltk import sent_tokenize

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError:
    # This block should now ideally not be reached if pip install worked
    print("Error: Required packages not found. Please check your installation.")
    sys.exit(1)

try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    print("Downloading NLTK punkt_tab...")
    nltk.download("punkt")
    nltk.download("punkt_tab")

def embed_sentence(embedder, sentence: str) -> np.ndarray:
    emb = embedder.encode(sentence, convert_to_numpy=True)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)

def main():
    print("📥 Loading models (MiniLM for FAISS, RoBERTa for Oracle)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") # FAISS uses CPU

    # We use GPU for RoBERTa if available to speed up the Oracle test
    try:
        nli_model = pipeline("text-classification", model="roberta-large-mnli", top_k=None, device=0)
    except Exception:
        print("CUDA not available for RoBERTa, falling back to CPU.")
        nli_model = pipeline("text-classification", model="roberta-large-mnli", top_k=None, device=-1)

    print("📥 Loading halueval-llm-spans dataset...")
    ds = load_dataset("llm-semantic-router/halueval-llm-spans", split="train")

    print("\n=======================================================")
    print(" 🔬 EXPERIMENT A: FAISS RETRIEVAL ACCURACY TEST")
    print("=======================================================\n")

    samples_tested = 0
    success_count = 0

    for row in ds:
        spans = row.get('labels', [])
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except Exception:
                continue

        # Find the first "Evident Conflict" in this sample
        conflict_span = None
        for span in spans:
            label = span.get('label', span.get('type', ''))
            if label == "Evident Conflict":
                conflict_span = span
                break

        if not conflict_span:
            continue # Skip if there are no direct conflicts

        # Extract the exact hallucinated string
        answer_text = str(row.get('answer', ''))
        start = conflict_span.get('start', conflict_span.get('start_char', 0))
        end = conflict_span.get('end', conflict_span.get('end_char', 0))
        target_lie = answer_text[start:end].strip()

        if not target_lie or len(target_lie) < 10:
            continue # Skip corrupted offsets

        samples_tested += 1

        # 1. Build the FAISS Index from the Source Prompt
        prompt_text = str(row.get('prompt', ''))
        # The dataset prompt includes "Summarize the following text...". We strip the instruction.
        if ":" in prompt_text[:100]:
            prompt_text = prompt_text.split(":", 1)[-1].strip()

        source_sentences = sent_tokenize(prompt_text)
        dim = embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)

        for sent in source_sentences:
            if len(sent.strip()) > 5:
                emb = embed_sentence(embedder, sent).reshape(1, -1)
                index.add(emb)
        # 2. Sweep over different K values to test Retrieval Depth (Recall@K)
        print(f"\n[{samples_tested}] ==========================================")
        print(f"🎯 TARGET LIE: \"{target_lie}\"")
        
        # Test how deep we have to dig to find the contradiction
        k_values_to_test = [1, 2, 3, 5, 10]
        
        # We only need to run FAISS once for the maximum k to get the top 10
        max_k = min(10, len(source_sentences))
        lie_emb = embed_sentence(embedder, target_lie).reshape(1, -1)
        D, I = index.search(lie_emb, max_k)
        
        # Store all retrieved sentences up to max_k
        retrieved_sentences = []
        for idx in I[0]:
            if idx >= 0 and idx < len(source_sentences):
                retrieved_sentences.append(source_sentences[idx])

        # 3. Oracle NLI Test: Evaluate cumulatively for each K
        for current_k in k_values_to_test:
            if current_k > len(retrieved_sentences):
                continue
                
            max_contradiction = 0.0
            best_sentence = ""
            
            # Evaluate only the top current_k sentences
            subset_to_evaluate = retrieved_sentences[:current_k]
            
            for context in subset_to_evaluate:
                try:
                    res = nli_model([{"text": context, "text_pair": target_lie}])
                    if isinstance(res, list) and len(res) > 0:
                        if isinstance(res[0], list):
                            res = res[0]
                        for d in res:
                            lab = str(d.get("label", "")).lower()
                            if "contradiction" in lab or lab == "label_0":
                                score = float(d.get("score", 0.0))
                                if score > max_contradiction:
                                    max_contradiction = score
                                    best_sentence = context
                except Exception:
                    continue
            
            # Print the result for this specific K threshold
            status = "✅ CAUGHT" if max_contradiction > 0.55 else "❌ MISSED"
            print(f"   ➤ K={current_k:<2} | {status} | Score: {max_contradiction:.3f} | Best Match: \"{best_sentence[:80]}...\"")
        
        print("-" * 70)
        
        print(f"\n[{samples_tested}] ==========================================")
        print("📜 FULL SOURCE TEXT (What FAISS had to choose from):")
        for i, sent in enumerate(source_sentences):
            print(f"  {i}: {sent}")
        print("---------------------------------------------------------")
        print(f"🎯 TARGET LIE: \"{target_lie}\"")
        for i, ctx in enumerate(retrieved_sentences):
            print(f"   📖 FAISS Retrieved #{i+1}: \"{ctx}\"")
        print(f"   🤖 RoBERTa Contradiction Score: {max_contradiction:.3f}")
        print(f"[{samples_tested}] 🎯 TARGET LIE: \"{target_lie}\"")
        for i, ctx in enumerate(retrieved_sentences):
            print(f"   📖 FAISS Retrieved #{i+1}: \"{ctx}\"")
        print(f"   🤖 RoBERTa Contradiction Score: {max_contradiction:.3f}")

        if max_contradiction > 0.55:
            print("   ✅ FAISS SUCCESS: Retrieved evidence proved the contradiction.")
            success_count += 1
        else:
            print("   ❌ FAISS FAILURE: Retrieved evidence was insufficient or neutral.")
        print("-" * 70)

        if samples_tested >= 10:
            break

    print(f"\n📊 FINAL FAISS Hits@2 ACCURACY (Small Sample): {success_count}/{samples_tested} ({success_count/samples_tested*100:.1f}%)")

if __name__ == "__main__":
    main()
    