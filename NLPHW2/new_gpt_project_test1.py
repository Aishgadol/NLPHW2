
import json
import math
import random
import os
import sys
import collections
import pandas as pd
from collections import Counter, defaultdict

def read_corpus(file_path):
    # this function reads the corpus from a jsonl file.
    # it expects each line to have 'protocol_type' and 'sentence_text' fields.
    # it returns two dataframes: one for committee sentences and one for plenary sentences.
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            protocol_type = obj['protocol_type'].strip().lower()
            sentence_text = obj['sentence_text'].strip()
            data.append((protocol_type, sentence_text))
    df = pd.DataFrame(data, columns=['protocol_type', 'sentence_text'])
    committee_df = df[df.protocol_type == 'committee'].copy()
    plenary_df = df[df.protocol_type == 'plenary'].copy()
    return committee_df, plenary_df

def tokenize_sentence(sentence):
    # this splits a sentence by whitespace into tokens
    return sentence.strip().split()

class TrigramLanguageModel:
    # this class builds a trigram language model with given interpolation lambdas
    # it uses laplace smoothing and linear interpolation of unigram, bigram, and trigram probabilities
    # special start tokens s_0 and s_1 are added to handle the first tokens in each sentence
    def __init__(self, sentences, lambdas):
        self.l1, self.l2, self.l3 = lambdas
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        # build counts for n-grams
        for s in sentences:
            tokens = ['s_0', 's_1'] + tokenize_sentence(s)
            self.unigram_counts.update(tokens)
            self.bigram_counts.update(zip(tokens[:-1], tokens[1:]))
            self.trigram_counts.update(zip(tokens[:-2], tokens[1:-1], tokens[2:]))

        self.vocab_size = len(self.unigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())

    def _laplace_smooth(self, count, context_count):
        # laplace smoothing: (count+1)/(context_count+vocab_size)
        return (count + 1) / (context_count + self.vocab_size)

    def calculate_prob_of_sentence(self, sentence):
        # calculates the log probability (base e) of a sentence
        # uses the trigram model with interpolation
        tokens = ['s_0', 's_1'] + tokenize_sentence(sentence)
        log_prob = 0.0
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            p_uni = self._laplace_smooth(self.unigram_counts[w3], self.total_unigrams)
            p_bi = self._laplace_smooth(self.bigram_counts.get((w2, w3), 0), self.unigram_counts.get(w2, 0))
            p_tri = self._laplace_smooth(self.trigram_counts.get((w1, w2), {}).get(w3, 0),
                                         self.bigram_counts.get((w1, w2), 0))
            prob = self.l1 * p_uni + self.l2 * p_bi + self.l3 * p_tri
            log_prob += math.log(prob)
        return log_prob

    def generate_next_token(self, context, is_final=False):
        # given a context, picks the token with highest probability
        # excludes 's_0' and 's_1' from predictions
        # if is_final is True and the chosen token is a comma, replace it with a period
        tokens = tokenize_sentence(context)
        if len(tokens) < 2:
            tokens = ['s_0', 's_1'] + tokens
        w1, w2 = tokens[-2], tokens[-1]

        max_token = None
        max_prob = -1.0
        for w3 in self.unigram_counts:
            if w3 in ('s_0', 's_1'):
                continue
            p_uni = (self.unigram_counts[w3] + 1) / (self.total_unigrams + self.vocab_size)
            p_bi = (self.bigram_counts.get((w2, w3), 0) + 1) / (self.unigram_counts.get(w2, 0) + self.vocab_size)
            p_tri = (self.trigram_counts.get((w1, w2), {}).get(w3, 0) + 1) / \
                    (self.bigram_counts.get((w1, w2), 0) + self.vocab_size)
            prob = self.l1 * p_uni + self.l2 * p_bi + self.l3 * p_tri
            if prob > max_prob:
                max_prob = prob
                max_token = w3

        # handle edge case: if final token is comma, change it to period
        if is_final and max_token == ',':
            max_token = '.'

        return max_token

def mask_tokens(sentences, mask_ratio):
    # masks a given ratio of tokens in each sentence with '[*]'
    # ensures at least one token is masked
    masked = []
    for s in sentences:
        tokens = tokenize_sentence(s)
        if len(tokens) == 0:
            masked.append(s)
            continue
        num_to_mask = max(1, int(len(tokens) * mask_ratio))
        if num_to_mask > len(tokens):
            num_to_mask = len(tokens)
        indices = random.sample(range(len(tokens)), num_to_mask)
        for i in indices:
            tokens[i] = '[*]'
        masked.append(' '.join(tokens))
    return masked

def get_k_n_t_collocations(k, n, t, corpus_df, metric_type):
    # extracts top k collocations of length n that appear at least t times
    # if metric_type == "frequency", returns top frequency-based collocations
    # if metric_type == "tfidf", uses a simple tf-idf measure
    docs = corpus_df["sentence_text"].tolist()
    ngram_counts = Counter()
    doc_counts = defaultdict(int)

    for doc in docs:
        tokens = tokenize_sentence(doc)
        ngs = list(zip(*[tokens[i:] for i in range(n)]))
        ngram_counts.update(ngs)
        unique_ngrams = set(ngs)
        for ng in unique_ngrams:
            doc_counts[ng] += 1

    if metric_type == "frequency":
        filtered = [(ngram, count) for ngram, count in ngram_counts.items() if count >= t]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:k]
    elif metric_type == "tfidf":
        total_docs = len(docs)
        total_terms = sum(ngram_counts.values())
        tfidf_scores = []
        for ngram, count in ngram_counts.items():
            if count >= t:
                tf = count / total_terms if total_terms > 0 else 0
                idf = math.log(total_docs / (1 + doc_counts[ngram]))
                tfidf_scores.append((ngram, tf * idf))
        return sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:k]

    return []

def write_collocations_to_file(collocations, file_path):
    # writes the collocations to a text file
    with open(file_path, 'w', encoding='utf-8') as f:
        for ngram_type, corpus_data in collocations.items():
            f.write(f"{ngram_type} collocations:\n")
            for corpus, ngrams in corpus_data.items():
                f.write(f"{corpus} corpus:\n")
                for ngram, score in ngrams:
                    ngram_str = ' '.join(ngram)
                    f.write(f"{ngram_str} ({score:.2f})\n")
                f.write("\n")

def write_sampled_sentences_to_files(original, masked, output_dir):
    # writes the original and masked sampled sentences to files
    with open(os.path.join(output_dir, 'sampled_sents_original.txt'), 'w', encoding='utf-8') as f:
        for sentence in original:
            f.write(sentence + '\n')

    with open(os.path.join(output_dir, 'sampled_sents_masked.txt'), 'w', encoding='utf-8') as f:
        for sentence in masked:
            f.write(sentence + '\n')

def write_results_to_file(results, file_path):
    # writes the prediction results to a file
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"original_sentence: {result['original']}\n")
            f.write(f"masked_sentence: {result['masked']}\n")
            f.write(f"plenary_sentence: {result['plenary']}\n")
            f.write(f"plenary_tokens: {', '.join(result['tokens'])}\n")
            f.write(f"probability of plenary sentence in plenary corpus: {result['prob_plenary_in_plenary']:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {result['prob_plenary_in_committee']:.2f}\n\n")

def compute_perplexity(model, masked_sentences, filled_sentences):
    # computes perplexity based on masked tokens only
    log_prob_sum = 0
    token_count = 0
    for masked, filled in zip(masked_sentences, filled_sentences):
        masked_tokens = tokenize_sentence(masked)
        filled_tokens = ['s_0', 's_1'] + tokenize_sentence(filled)
        for i, token in enumerate(masked_tokens):
            if token == '[*]':
                w1, w2, w3 = filled_tokens[i], filled_tokens[i + 1], filled_tokens[i + 2]
                p_uni = (model.unigram_counts[w3] + 1) / (model.total_unigrams + model.vocab_size)
                p_bi = (model.bigram_counts.get((w2, w3), 0) + 1) / (model.unigram_counts.get(w2, 0) + model.vocab_size)
                p_tri = (model.trigram_counts.get((w1, w2), {}).get(w3, 0) + 1) / (model.bigram_counts.get((w1, w2), 0) + model.vocab_size)
                prob = model.l1 * p_uni + model.l2 * p_bi + model.l3 * p_tri
                log_prob_sum += math.log(prob)
                token_count += 1
    if token_count == 0:
        return float('inf')
    return math.exp(-log_prob_sum / token_count)

def main(corpus_file, output_dir):
    # main pipeline
    # 1. read corpus
    committee_df, plenary_df = read_corpus(corpus_file)

    # 2. set seed for consistency
    random.seed(42)

    # 3. define lambdas
    lambdas = (0.04, 0.24, 0.72)

    # 4. build committee and plenary models
    committee_model = TrigramLanguageModel(committee_df['sentence_text'], lambdas)
    plenary_model = TrigramLanguageModel(plenary_df['sentence_text'], lambdas)

    # 5. extract collocations
    collocations = {
        "Two-gram": {
            "Committee": get_k_n_t_collocations(10, 2, 5, committee_df, "frequency"),
            "Plenary": get_k_n_t_collocations(10, 2, 5, plenary_df, "frequency")
        },
        "Three-gram": {
            "Committee": get_k_n_t_collocations(10, 3, 5, committee_df, "tfidf"),
            "Plenary": get_k_n_t_collocations(10, 3, 5, plenary_df, "tfidf")
        },
        "Four-gram": {
            "Committee": get_k_n_t_collocations(10, 4, 5, committee_df, "frequency"),
            "Plenary": get_k_n_t_collocations(10, 4, 5, plenary_df, "frequency")
        }
    }
    write_collocations_to_file(collocations, os.path.join(output_dir, 'knesset_collocations.txt'))

    # 6. sample 10 committee sentences
    sample_sentences = random.sample(committee_df['sentence_text'].tolist(), 10)
    # 7. mask 10% of tokens
    masked_sentences = mask_tokens(sample_sentences, 0.1)
    write_sampled_sentences_to_files(sample_sentences, masked_sentences, output_dir)

    # 8. fill masked tokens using plenary model
    filled_sentences = []
    results = []
    for masked, original in zip(masked_sentences, sample_sentences):
        tokens = masked.split()
        filled = tokens[:]
        generated_tokens = []
        for i, token in enumerate(tokens):
            if token == '[*]':
                context = ' '.join(filled[:i])
                # check if final token
                is_final_token = (i == len(tokens)-1)
                next_token = plenary_model.generate_next_token(context, is_final=is_final_token)
                filled[i] = next_token
                generated_tokens.append(next_token)
        filled_sentence = ' '.join(filled)
        filled_sentences.append(filled_sentence)
        results.append({
            'original': original,
            'masked': masked,
            'plenary': filled_sentence,
            'tokens': generated_tokens,
            'prob_plenary_in_plenary': plenary_model.calculate_prob_of_sentence(filled_sentence),
            'prob_plenary_in_committee': committee_model.calculate_prob_of_sentence(filled_sentence)
        })

    write_results_to_file(results, os.path.join(output_dir, 'sampled_sents_results.txt'))

    # 9. compute perplexity
    perplexity = compute_perplexity(plenary_model, masked_sentences, filled_sentences)
    with open(os.path.join(output_dir, 'result_perplexity.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{perplexity:.2f}\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python knesset_language_models.py <path_to_corpus> <output_dir>")
        sys.exit(1)
    corpus_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(corpus_file, output_dir)
