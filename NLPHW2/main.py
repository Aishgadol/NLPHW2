import json
import math
import random
import os
import sys
import collections
import pandas as pd
import re

"""
More updates:
When tokenizing a sentence, we must ignore every symbol that is not a Hebrew word.
We will:
- Use a regex to extract only sequences of Hebrew letters as tokens.
- A Hebrew word is defined by the regex HEBREW_PATTERN = r'[\u0590-\u05FF]+'

For example:
"המכונית הגדולה - שלי , אוהבת כלבים"
Should tokenize into: ["המכונית", "הגדולה", "שלי", "אוהבת", "כלבים"]
All punctuation or non-Hebrew sequences are ignored.

We retain the previous constraints:
- Predict only Hebrew words (already ensured by is_valid_word and the model).
- We use linear interpolation and smoothing as before.

DO NOT FAIL ME
"""

HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]+$')
HEBREW_FINDER = re.compile(r'[\u0590-\u05FF]+')

def is_valid_word(w):
    # Must be Hebrew only word:
    # matched fully by HEBREW_PATTERN
    if w.startswith('_'):
        return False
    # Check if it fully matches Hebrew letters pattern:
    if HEBREW_PATTERN.match(w):
        return True
    return False

class LM_Trigram:
    def __init__(self, sentences, lambdas=(1/3,1/3,1/3)):
        print("Initializing LM_Trigram with lambdas:", lambdas)
        self.l1, self.l2, self.l3 = lambdas
        self._build_counts(sentences)
        print("LM_Trigram initialized. Vocabulary size:", self.V, "Total tokens:", self.total_unigrams)

    def _build_counts(self, sentences):
        print("Building n-gram counts...")
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()

        for sent in sentences:
            s = ["_s0_", "_s1_"] + sent
            for w in s:
                self.unigram_counts[w] += 1
            for i in range(len(s)-1):
                self.bigram_counts[(s[i], s[i+1])] += 1
            for i in range(len(s)-2):
                self.trigram_counts[(s[i], s[i+1], s[i+2])] += 1

        self.V = len(self.unigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())

        print("Counts built. Unigrams:", len(self.unigram_counts),
              "Bigrams:", len(self.bigram_counts),
              "Trigrams:", len(self.trigram_counts))

    def _laplace_unigram(self, w):
        return (self.unigram_counts[w] + 1) / (self.total_unigrams + self.V)

    def _laplace_bigram(self, w1, w2):
        count_bigram = self.bigram_counts[(w1,w2)]
        count_context = self.unigram_counts[w1]
        return (count_bigram + 1) / (count_context + self.V)

    def _laplace_trigram(self, w1, w2, w3):
        count_trigram = self.trigram_counts[(w1,w2,w3)]
        count_context = self.bigram_counts[(w1,w2)]
        return (count_trigram + 1) / (count_context + self.V)

    def _interp_prob(self, w1, w2, w3):
        p_uni = self._laplace_unigram(w3)
        p_bi  = self._laplace_bigram(w2, w3)
        p_tri = self._laplace_trigram(w1, w2, w3)
        return self.l1*p_uni + self.l2*p_bi + self.l3*p_tri

    def calculate_prob_of_sentence(self, sentence):
        tokens = sentence.strip().split()
        tokens = ["_s0_", "_s1_"] + tokens
        log_prob = 0.0
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            p = self._interp_prob(w1, w2, w3)
            log_prob += math.log(p)
        return log_prob

    def generate_next_token(self, context):
        context_tokens = context.strip().split()
        if len(context_tokens) < 2:
            context_tokens = ["_s0_", "_s1_"] + context_tokens
        w1, w2 = context_tokens[-2:] if len(context_tokens)>=2 else ("_s0_","_s1_")

        max_token = None
        max_prob = -1
        for w3 in self.unigram_counts.keys():
            if not is_valid_word(w3):
                continue
            p = self._interp_prob(w1, w2, w3)
            if p > max_prob:
                max_prob = p
                max_token = w3
        return (max_token, max_prob)


def read_corpus(jsonl_file):
    print("Reading corpus from:", jsonl_file)
    data = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    t = obj["type_protocol"].strip()
                    if t == "ועדה":
                        t = "committee"
                    elif t == "מליאה":
                        t = "plenary"
                    txt = obj["text_sentence"].strip()
                    data.append((t, txt))
    except Exception as e:
        print("Error reading JSONL:", e)
        sys.exit(1)

    df = pd.DataFrame(data, columns=["type_protocol","text_sentence"])
    print("Finished reading corpus.")
    committee_df = df[df.type_protocol == "committee"].copy()
    plenary_df = df[df.type_protocol == "plenary"].copy()
    print(f"Committee sentences: {len(committee_df)}, Plenary sentences: {len(plenary_df)}")
    return committee_df, plenary_df

def tokenize_sentence(sent):
    # Extract only Hebrew words from the sentence
    # Using HEBREW_FINDER to find all Hebrew sequences
    tokens = re.findall(HEBREW_FINDER, sent)
    return tokens

def build_model(df, lambdas=(1/3,1/3,1/3)):
    print("Building model...")
    sentences = []
    for txt in df["text_sentence"]:
        tokens = tokenize_sentence(txt)
        sentences.append(tokens)
    model = LM_Trigram(sentences, lambdas=lambdas)
    print("Model built.")
    return model

def get_k_n_t_collocations(k, n, t, corpus_df, metric_type="frequency"):
    docs = corpus_df["text_sentence"].tolist()
    ngram_doc_freq = {}
    for i, doc in enumerate(docs):
        tokens = tokenize_sentence(doc)
        ngs = list(zip(*[tokens[j:] for j in range(n)]))
        doc_count = collections.Counter(ngs)
        for ng, c in doc_count.items():
            if ng not in ngram_doc_freq:
                ngram_doc_freq[ng] = {}
            ngram_doc_freq[ng][i] = c
    ngram_global_count = {ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
    filtered_ngrams = {ng: freq for ng, freq in ngram_global_count.items() if freq >= t}

    if metric_type == "frequency":
        sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x:x[1], reverse=True)
        return [(" ".join(ng), freq) for ng, freq in sorted_ngrams[:k]]
    elif metric_type == "tfidf":
        D = len(docs)
        if D == 0:
            return []
        doc_count_for_t = {ng: len(ngram_doc_freq[ng]) for ng in filtered_ngrams}
        doc_all_ngrams_counts = []
        for d_i, doc in enumerate(docs):
            tokens = tokenize_sentence(doc)
            doc_ngs = list(zip(*[tokens[j:] for j in range(n)]))
            doc_all_ngrams_counts.append(collections.Counter(doc_ngs))

        tfidf_scores = {}
        for ng in filtered_ngrams:
            if doc_count_for_t[ng] == 0:
                continue
            idf = math.log(D / doc_count_for_t[ng])
            score = 0.0
            for d_i, freq_map in ngram_doc_freq[ng].items():
                f_t_d = freq_map
                total_terms = sum(doc_all_ngrams_counts[d_i].values())
                tf = f_t_d / total_terms if total_terms > 0 else 0
                score += tf * idf
            tfidf_scores[ng] = score
        sorted_ngrams = sorted(tfidf_scores.items(), key=lambda x:x[1], reverse=True)
        return [(" ".join(ng), sc) for ng, sc in sorted_ngrams[:k]]
    else:
        return []

def mask_tokens_in_sentences(sentences, x):
    print("Masking tokens in sampled sentences...")
    masked_sents = []
    for sent in sentences:
        tokens = tokenize_sentence(sent)
        if len(tokens) == 0:
            # If no Hebrew tokens, just append empty
            masked_sents.append("")
            continue
        num_to_mask = max(1, int(len(tokens)*x/100))
        if num_to_mask > len(tokens):
            num_to_mask = len(tokens)
        indices = random.sample(range(len(tokens)), num_to_mask)
        for idx in indices:
            tokens[idx] = "[*]"
        masked_sents.append(" ".join(tokens))
    print("Masking complete.")
    return masked_sents

def fill_masked_tokens(masked_sentence, lm):
    tokens = masked_sentence.strip().split()
    predicted_tokens = []
    # Find fallback word if needed
    valid_hebrew_words = [w for w in lm.unigram_counts if is_valid_word(w)]
    fallback_word = None
    if valid_hebrew_words:
        fallback_word = max(valid_hebrew_words, key=lambda w: lm.unigram_counts[w])
    else:
        fallback_word = "שלום"

    for i, tk in enumerate(tokens):
        if tk == "[*]":
            context = " ".join(tokens[:i])
            pred_token, pred_prob = lm.generate_next_token(context)
            if pred_token is None:
                pred_token = fallback_word
            tokens[i] = pred_token
            predicted_tokens.append(pred_token)
    return " ".join(tokens), predicted_tokens

def compute_sentence_prob(lm, sentence):
    return lm.calculate_prob_of_sentence(sentence)

def compute_perplexity_of_masked_tokens(lm, original_masked_sentences, filled_sentences):
    print("Computing perplexity for masked tokens...")
    log_probs = []
    total_masked_tokens = 0

    for (original_masked, filled) in zip(original_masked_sentences, filled_sentences):
        # original_masked and filled are tokenized differently now
        # We must re-tokenize original_masked as well for indexing
        orig_tokens = original_masked.strip().split()
        filled_tokens = filled.strip().split()
        # Add start tokens
        f_tokens = ["_s0_", "_s1_"] + filled_tokens
        for i, tk in enumerate(orig_tokens):
            if tk == "[*]":
                filled_pos = i+2
                if filled_pos < len(f_tokens):
                    w1, w2, w3 = f_tokens[filled_pos-2], f_tokens[filled_pos-1], f_tokens[filled_pos]
                    p = lm._interp_prob(w1, w2, w3)
                    log_p = math.log(p, 2)
                    log_probs.append(log_p)
                    total_masked_tokens += 1

    if total_masked_tokens == 0:
        print("No masked tokens found, perplexity = 0.00")
        return 0.0
    avg_log_p = sum(log_probs)/total_masked_tokens
    pp = 2 ** (-avg_log_p)
    print(f"Perplexity computed: {pp:.2f}")
    return pp

def main():
    jsonl_file = "result.jsonl"
    if not os.path.exists(jsonl_file):
        print("result.jsonl not found.")
        sys.exit(1)
    print("Starting main process...")

    committee_df, plenary_df = read_corpus(jsonl_file)

    print("Building committee model...")
    committee_model = build_model(committee_df, lambdas=(1/3,1/3,1/3))
    print("Building plenary model...")
    plenary_model = build_model(plenary_df, lambdas=(1/3,1/3,1/3))

    print("Extracting and writing collocations...")
    with open("collocations_knesset.txt", "w", encoding="utf-8") as f:
        for n in [2,3,4]:
            if n == 2:
                f.write("Two-gram collocations:\n")
            elif n == 3:
                f.write("Three-gram collocations:\n")
            else:
                f.write("Four-gram collocations:\n")

            f.write("Frequency:\n")
            f.write("Committee corpus:\n")
            freq_comm = get_k_n_t_collocations(10, n, 5, committee_df, "frequency")
            for coll, val in freq_comm:
                f.write(f"{coll}\n")
            f.write("\n")

            f.write("Plenary corpus:\n")
            freq_plen = get_k_n_t_collocations(10, n, 5, plenary_df, "frequency")
            for coll, val in freq_plen:
                f.write(f"{coll}\n")
            f.write("\n")

            f.write("TF-IDF:\n")
            f.write("Committee corpus:\n")
            tfidf_comm = get_k_n_t_collocations(10, n, 5, committee_df, "tfidf")
            for coll, val in tfidf_comm:
                f.write(f"{coll}\n")
            f.write("\n")

            f.write("Plenary corpus:\n")
            tfidf_plen = get_k_n_t_collocations(10, n, 5, plenary_df, "tfidf")
            for coll, val in tfidf_plen:
                f.write(f"{coll}\n")
            f.write("\n")

    committee_sents = committee_df["text_sentence"].tolist()
    if len(committee_sents) < 10:
        print("Not enough committee sentences to sample 10. Sampling fewer:", len(committee_sents))
    sample_size = min(10, len(committee_sents))
    if sample_size == 0:
        print("No committee sentences available. Exiting.")
        sys.exit(1)

    random.seed(41)
    sampled_sents_raw = random.sample(committee_sents, sample_size)
    # We must store original tokenizations and masked versions
    # Original tokenization is done by tokenize_sentence, so let's keep original as is
    # but we also apply tokenize to them when masking.
    sampled_sents = [" ".join(tokenize_sentence(s)) for s in sampled_sents_raw]
    masked_sents = mask_tokens_in_sentences(sampled_sents, 15)

    print("Writing sampled original and masked sentences...")
    with open("sents_sampled_original.txt", "w", encoding="utf-8") as f:
        for s in sampled_sents:
            f.write(s+"\n")

    with open("sents_sampled_masked.txt", "w", encoding="utf-8") as f:
        for s in masked_sents:
            f.write(s+"\n")

    print("Filling masked tokens using plenary model and computing probabilities...")
    results = []
    for orig_sent, masked_sent in zip(sampled_sents, masked_sents):
        filled_sentence, predicted_tokens = fill_masked_tokens(masked_sent, plenary_model)
        prob_plenary_in_plenary = compute_sentence_prob(plenary_model, filled_sentence)
        prob_plenary_in_committee = compute_sentence_prob(committee_model, filled_sentence)

        results.append({
            "original_sentence": orig_sent,
            "masked_sentence": masked_sent,
            "plenary_sentence": filled_sentence,
            "plenary_tokens": ",".join(predicted_tokens),
            "prob_pp": prob_plenary_in_plenary,
            "prob_pc": prob_plenary_in_committee
        })

    print("Writing results of masked sentence completions...")
    with open("results_sents_sampled.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"original_sentence: {r['original_sentence']}\n")
            f.write(f"masked_sentence: {r['masked_sentence']}\n")
            f.write(f"plenary_sentence: {r['plenary_sentence']}\n")
            f.write(f"plenary_tokens: {r['plenary_tokens']}\n")
            f.write(f"probability of plenary sentence in plenary corpus: {r['prob_pp']:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {r['prob_pc']:.2f}\n")

    # For perplexity, we must consider only masked tokens from original_masked_sents as we did:
    # We have masked_sents and filled_sents for perplexity calculation.
    pp = compute_perplexity_of_masked_tokens(plenary_model, masked_sents, [r['plenary_sentence'] for r in results])
    with open("result_perplexity.txt", "w", encoding="utf-8") as f:
        f.write(f"{pp:.2f}\n")

    print("Process completed successfully. Check result_perplexity.txt for perplexity.")

if __name__ == "__main__":
    main()
