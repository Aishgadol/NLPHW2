import json
import math
import random
import os
import sys
import collections
import itertools
import numpy as np
import pandas as pd

"""
Updated code:
- Converts "ועדה" to "committee" and "מליאה" to "plenary" when reading the JSON.
- Adds printing statements throughout the process to indicate progress.
"""


class LM_Trigram:
    def __init__(self, sentences, lambdas=(0.1, 0.3, 0.6)):
        print("Initializing LM_Trigram...")
        self.l1, self.l2, self.l3 = lambdas
        self._build_counts(sentences)
        print("LM_Trigram initialized.")

    def _build_counts(self, sentences):
        print("Building n-gram counts...")
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()

        for sent in sentences:
            s = ["_s0_", "_s1_"] + sent
            for w in s:
                self.unigram_counts[w] += 1
            for i in range(len(s) - 1):
                self.bigram_counts[(s[i], s[i + 1])] += 1
            for i in range(len(s) - 2):
                self.trigram_counts[(s[i], s[i + 1], s[i + 2])] += 1

        self.V = len(self.unigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())

        self.bigram_context_counts = collections.Counter()
        for (w1, w2), c in self.bigram_counts.items():
            self.bigram_context_counts[w1] += c

        self.trigram_context_counts = collections.Counter()
        for (w1, w2, w3), c in self.trigram_counts.items():
            self.trigram_context_counts[(w1, w2)] += c
        print("Counts built.")

    def _laplace_unigram(self, w):
        return (self.unigram_counts[w] + 1) / (self.total_unigrams + self.V)

    def _laplace_bigram(self, w1, w2):
        count_bigram = self.bigram_counts[(w1, w2)]
        count_context = self.unigram_counts[w1]
        return (count_bigram + 1) / (count_context + self.V)

    def _laplace_trigram(self, w1, w2, w3):
        count_trigram = self.trigram_counts[(w1, w2, w3)]
        count_context = self.bigram_counts[(w1, w2)]
        return (count_trigram + 1) / (count_context + self.V)

    def _interp_prob(self, w1, w2, w3):
        p_uni = self._laplace_unigram(w3)
        p_bi = self._laplace_bigram(w2, w3)
        p_tri = self._laplace_trigram(w1, w2, w3)
        return self.l1 * p_uni + self.l2 * p_bi + self.l3 * p_tri

    def calculate_prob_of_sentence(self, sentence):
        tokens = sentence.strip().split()
        tokens = ["_s0_", "_s1_"] + tokens
        log_prob = 0.0
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i - 2], tokens[i - 1], tokens[i]
            p = self._interp_prob(w1, w2, w3)
            log_prob += math.log(p)
        return log_prob

    def generate_next_token(self, context):
        context_tokens = context.strip().split()
        if len(context_tokens) < 2:
            context_tokens = ["_s0_", "_s1_"] + context_tokens
        w1, w2 = context_tokens[-2:] if len(context_tokens) >= 2 else ("_s0_", "_s1_")

        max_token = None
        max_prob = -1
        for w3 in self.unigram_counts.keys():
            p = self._interp_prob(w1, w2, w3)
            if p > max_prob:
                max_prob = p
                max_token = w3
        return (max_token, max_prob)


def read_corpus(jsonl_file):
    print("Reading corpus from:", jsonl_file)
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # Convert Hebrew to English type:
                # "ועדה" => "committee"
                # "מליאה" => "plenary"
                t = obj["protocol_type"].strip()
                if t == "committee":
                    t = "committee"
                elif t == "plenary":
                    t = "plenary"
                txt = obj["sentence_text"].strip()
                data.append((t, txt))
    df = pd.DataFrame(data, columns=["protocol_type", "sentence_text"])
    print("Finished reading corpus.")
    committee_df = df[df.protocol_type == "committee"].copy()
    plenary_df = df[df.protocol_type == "plenary"].copy()
    print(f"Committee sentences: {len(committee_df)}, Plenary sentences: {len(plenary_df)}")
    return committee_df, plenary_df


def tokenize_sentence(sent):
    return sent.strip().split()


def build_model(df):
    print("Building model...")
    sentences = []
    for txt in df["sentence_text"]:
        tokens = tokenize_sentence(txt)
        sentences.append(tokens)
    model = LM_Trigram(sentences)
    print("Model built.")
    return model


def get_k_n_t_collocations(k, n, t, corpus_df, metric_type="frequency"):
    docs = corpus_df["sentence_text"].tolist()
    ngram_doc_freq = {}
    for i, doc in enumerate(docs):
        tokens = tokenize_sentence(doc)
        ngs = zip(*[tokens[j:] for j in range(n)])
        doc_count = collections.Counter(ngs)
        for ng, c in doc_count.items():
            if ng not in ngram_doc_freq:
                ngram_doc_freq[ng] = {}
            ngram_doc_freq[ng][i] = c
    ngram_global_count = {ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
    filtered_ngrams = {ng: freq for ng, freq in ngram_global_count.items() if freq >= t}

    if metric_type == "frequency":
        sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: x[1], reverse=True)
        return [(" ".join(ng), freq) for ng, freq in sorted_ngrams[:k]]
    elif metric_type == "tfidf":
        D = len(docs)
        doc_count_for_t = {ng: len(ngram_doc_freq[ng]) for ng in filtered_ngrams}
        doc_all_ngrams_counts = []
        for d_i, doc in enumerate(docs):
            tokens = tokenize_sentence(doc)
            doc_ngs = list(zip(*[tokens[j:] for j in range(n)]))
            doc_all_ngrams_counts.append(collections.Counter(doc_ngs))

        tfidf_scores = {}
        for ng in filtered_ngrams:
            idf = math.log(D / doc_count_for_t[ng])
            score = 0.0
            for d_i, freq_map in ngram_doc_freq[ng].items():
                f_t_d = freq_map
                total_terms = sum(doc_all_ngrams_counts[d_i].values())
                tf = f_t_d / total_terms
                score += tf * idf
            tfidf_scores[ng] = score
        sorted_ngrams = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(" ".join(ng), sc) for ng, sc in sorted_ngrams[:k]]
    else:
        return []


def mask_tokens_in_sentences(sentences, x):
    print("Masking tokens in sampled sentences...")
    masked_sents = []
    for sent in sentences:
        tokens = tokenize_sentence(sent)
        if len(tokens) == 0:
            masked_sents.append(sent)
            continue
        num_to_mask = max(1, int(len(tokens) * x / 100))
        indices = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))
        for idx in indices:
            tokens[idx] = "[*]"
        masked_sents.append(" ".join(tokens))
    print("Masking complete.")
    return masked_sents


def fill_masked_tokens(masked_sentence, lm):
    tokens = tokenize_sentence(masked_sentence)
    predicted_tokens = []
    for i, tk in enumerate(tokens):
        if tk == "[*]":
            context = " ".join(tokens[:i])
            pred_token, pred_prob = lm.generate_next_token(context)
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
        orig_tokens = tokenize_sentence(original_masked)
        filled_tokens = tokenize_sentence(filled)
        f_tokens = ["_s0_", "_s1_"] + filled_tokens
        for i, tk in enumerate(orig_tokens):
            if tk == "[*]":
                filled_pos = i + 2
                w1, w2, w3 = f_tokens[filled_pos - 2], f_tokens[filled_pos - 1], f_tokens[filled_pos]
                p = lm._interp_prob(w1, w2, w3)
                log_p = math.log(p, 2)
                log_probs.append(log_p)
                total_masked_tokens += 1

    if total_masked_tokens == 0:
        print("No masked tokens found, perplexity = 0")
        return 0.0
    avg_log_p = sum(log_probs) / total_masked_tokens
    pp = 2 ** (-avg_log_p)
    print("Perplexity computed.")
    return pp


def main():
    jsonl_file = "result.jsonl"
    if not os.path.exists(jsonl_file):
        print("result.jsonl not found.")
        sys.exit(1)
    print("Starting main process...")
    committee_df, plenary_df = read_corpus(jsonl_file)

    print("Building committee model...")
    committee_model = build_model(committee_df)
    print("Building plenary model...")
    plenary_model = build_model(plenary_df)

    with open("collocations_knesset.txt", "w", encoding="utf-8") as f:
        for n in [2, 3, 4]:
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

    committee_sents = committee_df["sentence_text"].tolist()
    if len(committee_sents) < 10:
        print("Not enough committee sentences to sample. Will sample fewer.")
    sample_size = min(10, len(committee_sents))
    if sample_size == 0:
        raise ValueError("No committee sentences available.")

    random.seed(123)
    sampled_sents = random.sample(committee_sents, sample_size)
    masked_sents = mask_tokens_in_sentences(sampled_sents, 10)

    with open("sents_sampled_original.txt", "w", encoding="utf-8") as f:
        for s in sampled_sents:
            f.write(s + "\n")

    with open("sents_sampled_masked.txt", "w", encoding="utf-8") as f:
        for s in masked_sents:
            f.write(s + "\n")

    print("Filling masked tokens using plenary model...")
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

    with open("results_sents_sampled.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"original_sentence: {r['original_sentence']}\n")
            f.write(f"masked_sentence: {r['masked_sentence']}\n")
            f.write(f"plenary_sentence: {r['plenary_sentence']}\n")
            f.write(f"plenary_tokens: {r['plenary_tokens']}\n")
            f.write(f"probability of plenary sentence in plenary corpus: {r['prob_pp']:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {r['prob_pc']:.2f}\n")

    filled_sents = [r['plenary_sentence'] for r in results]
    pp = compute_perplexity_of_masked_tokens(plenary_model, masked_sents, filled_sents)
    with open("result_perplexity.txt", "w", encoding="utf-8") as f:
        f.write(f"{pp:.2f}\n")

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
