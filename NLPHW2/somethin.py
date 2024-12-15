import collections
import json
import math
import random

def read_jsonl(file_path):
    """
    reads a jsonl file and separates sentences into committee and plenary matrices
    """
    try:
        word_matrix_c = []
        word_matrix_p = []
        # open the JSONL file
        with open(file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # assume `protocol_type` is either "committee" or "plenary"
                # and `sentence_text` is the tokenized sentence
                protocol_type = obj.get('protocol_type', '').strip().lower()
                sentence = obj.get('sentence_text', '').strip()
                if sentence and protocol_type in ['committee', 'plenary']:
                    words = sentence.split()
                    if protocol_type == 'plenary':
                        word_matrix_p.append(words)
                    else:
                        word_matrix_c.append(words)
        return word_matrix_c, word_matrix_p
    except Exception as error:
        print(f"Error occurred while reading from JSONL '{file_path}': {error}")
        return [], []

class Trigram_LM:
    """
    trigram language model with laplace smoothing and linear interpolation
    """
    def __init__(self, word_matrix):
        self.word_matrix = word_matrix
        self.trigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.bigram_counts = collections.defaultdict(int)
        self.unigram_counts = collections.defaultdict(int)
        self.N = 0
        self.V = 0
        self.analyze_matrix()

    def analyze_matrix(self):
        """
        counts unigrams, bigrams, and trigrams from the word matrix
        """
        mat = self.word_matrix
        mat = [word for sentence in mat for word in sentence]
        self.N = len(mat)
        for i in range(2, len(mat)):
            w1, w2, w3 = mat[i - 2], mat[i - 1], mat[i]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.bigram_counts[(w1, w2)] += 1
            self.unigram_counts[w3] += 1
        self.V = len(self.unigram_counts)

    def calculate_prob_of_sentence(self, tokens, smoothing='Laplace', lambdas=[0.8999999, 0.1, 0.0]):
        """
        calculates log probability of a given sentence using the trigram model
        """
        tokens = tokens.split()
        sol = 0
        if self.V == 0 or len(tokens) == 0:
            return -math.inf
        elif len(tokens) == 1:
            sol = math.log2((self.unigram_counts[tokens[0]] + 1) / (self.N + self.V))
        elif len(tokens) == 2:
            bigram_count = self.bigram_counts.get((tokens[0], tokens[1]), 0)
            if smoothing == 'Laplace':
                sol = math.log2((bigram_count + 1) / (self.unigram_counts.get(tokens[0], 0) + self.V))
            else:
                sol = math.log2(
                    lambdas[0] * (bigram_count + 1) / (self.unigram_counts.get(tokens[0], 0) + self.V) +
                    lambdas[1] * (self.unigram_counts.get(tokens[1], 0) + 1) / (self.N + self.V)
                )
        else:
            log_val = 0
            for i in range(2, len(tokens)):
                w1, w2, w3 = tokens[i - 2], tokens[i - 1], tokens[i]
                trigram_count = self.trigram_counts[(w1, w2)].get(w3, 0)
                bigram_count = self.bigram_counts.get((w1, w2), 0)
                if smoothing == 'Laplace':
                    smoothed_probability = (trigram_count + 1) / (bigram_count + self.V)
                else:
                    trigram_prob = (trigram_count + 1) / (bigram_count + self.V)
                    bigram_prob = (self.bigram_counts.get((w2, w3), 0) + 1) / (self.unigram_counts.get(w2, 0) + self.V)
                    unigram_prob = (self.unigram_counts.get(w3, 0) + 1) / (self.N + self.V)
                    smoothed_probability = lambdas[0] * trigram_prob + lambdas[1] * bigram_prob + lambdas[2] * unigram_prob
                log_val += math.log2(smoothed_probability)
            sol = log_val
        return sol

    def generate_next_token(self, tokens, comma_penalty=0.5, smoothing='Laplace', lambdas=[0.8999999, 0.1, 0.0]):
        """
        predicts the next token given the context
        applies a penalty to commas
        excludes '_s0_' and '_s1_' tokens from prediction
        """
        tokens = tokens.split()
        if len(tokens) < 2:
            tokens = ["_s0_", "_s1_"] + tokens
        w1, w2 = tokens[-2], tokens[-1]
        max_token = None
        max_prob = -1
        for w3 in self.unigram_counts:
            if w3 in ("_s0_", "_s1_"):
                continue  # exclude special tokens
            prob = self.calculate_prob_of_sentence(' '.join([w1, w2, w3]), smoothing, lambdas)
            # apply comma penalty
            if w3 == ',':
                prob += math.log2(comma_penalty)
            if prob > max_prob:
                max_prob = prob
                max_token = w3
        return max_token

    def get_k_n_collocations(self, k, n):
        """
        returns top k collocations of length n based on PMI
        """
        collocations = {}
        for sentence in self.word_matrix:
            for index in range(n, len(sentence)):
                words = tuple(sentence[index - n: index])
                if words not in collocations:
                    if n == 2:
                        # calculate PMI for bigrams
                        w1, w2 = words
                        p_w1 = self.unigram_counts.get(w1, 0) / self.N
                        p_w2 = self.unigram_counts.get(w2, 0) / self.N
                        p_w1_w2 = self.bigram_counts.get((w1, w2), 0) / self.N
                        if p_w1 > 0 and p_w2 > 0:
                            pmi = math.log2(p_w1_w2 / (p_w1 * p_w2))
                        else:
                            pmi = 0
                        collocations[words] = pmi
                    elif n == 3:
                        # calculate PMI for trigrams
                        w1, w2, w3 = words
                        p_w1 = self.unigram_counts.get(w1, 0) / self.N
                        p_w2 = self.unigram_counts.get(w2, 0) / self.N
                        p_w3 = self.unigram_counts.get(w3, 0) / self.N
                        p_w1_w2 = self.bigram_counts.get((w1, w2), 0) / self.N
                        p_w2_w3 = self.bigram_counts.get((w2, w3), 0) / self.N
                        p_w1_w2_w3 = self.trigram_counts.get((w1, w2), {}).get(w3, 0) / self.N
                        denom = p_w1 * p_w2 * p_w3
                        if denom > 0:
                            pmi = math.log2(p_w1_w2_w3 / denom) if p_w1_w2_w3 > 0 else 0
                        else:
                            pmi = 0
                        collocations[words] = pmi
                    elif n == 4:
                        # extend PMI calculation for four-grams if needed
                        # for simplicity, we'll skip PMI for four-grams
                        collocations[words] = 0
        # sort collocations by PMI descending
        sorted_collocations = sorted(collocations.items(), key=lambda x: x[1], reverse=True)
        return sorted_collocations[:k]

    def fill_tokens(self, sentence, smoothing='Laplace', lambdas=[0.8999999, 0.1, 0.0], comma_penalty=0.5):
        """
        fills the masked tokens in a sentence and returns the filled sentence
        """
        partial_sentence = sentence.copy()
        changed_words = []
        for i, word in enumerate(partial_sentence):
            if word == '[*]':
                context = ' '.join(partial_sentence[:i])
                predicted_word = self.generate_next_token(context, comma_penalty, smoothing, lambdas)
                partial_sentence[i] = predicted_word
                changed_words.append(predicted_word)
        return changed_words, partial_sentence

def write_txt_collocations(file_path, lists):
    """
    writes collocations to a text file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for collocation_type, corpus_collocations in lists.items():
                file.write(f"{collocation_type} collocations:\n")
                for corpus, collocation_list in corpus_collocations.items():
                    file.write(f"{corpus} corpus:\n")
                    for collocation_tuple in collocation_list:
                        collocation, pmi_value = collocation_tuple
                        formatted_collocation = ' '.join(collocation)
                        file.write(f"{formatted_collocation}\n")
                    file.write('\n')
    except Exception as error:
        print(f"Error occurred while writing to TXT '{file_path}': {error}")

def write_generated_sentences_to_txt(file_path, original_sentences, committee_lm, plenary_lm, comma_penalty):
    """
    writes the original and filled sentences along with probabilities to a text file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for original_sentence in original_sentences:
                file.write(f"Original sentence: {' '.join(original_sentence)}\n")

                # fill tokens using committee model
                committee_changed, committee_filled = committee_lm.fill_tokens(
                    original_sentence.copy(),
                    comma_penalty=comma_penalty
                )
                committee_sentence_str = ' '.join(committee_filled)
                committee_tokens_str = ','.join(committee_changed)
                committee_prob_in_committee = committee_lm.calculate_prob_of_sentence(
                    committee_sentence_str, 'Laplace'
                )
                committee_prob_in_plenary = plenary_lm.calculate_prob_of_sentence(
                    committee_sentence_str, 'Laplace'
                )

                file.write(f"Committee sentence: {committee_sentence_str}\n")
                file.write(f"Committee tokens: {committee_tokens_str}\n")
                file.write(f"Probability of committee sentence in committee corpus: {round(committee_prob_in_committee, 2)}\n")
                file.write(f"Probability of committee sentence in plenary corpus: {round(committee_prob_in_plenary, 2)}\n")

                # fill tokens using plenary model
                plenary_changed, plenary_filled = plenary_lm.fill_tokens(
                    original_sentence.copy(),
                    comma_penalty=comma_penalty
                )
                plenary_sentence_str = ' '.join(plenary_filled)
                plenary_tokens_str = ','.join(plenary_changed)
                plenary_prob_in_plenary = plenary_lm.calculate_prob_of_sentence(
                    plenary_sentence_str, 'Laplace'
                )
                plenary_prob_in_committee = committee_lm.calculate_prob_of_sentence(
                    plenary_sentence_str, 'Laplace'
                )

                file.write(f"Plenary sentence: {plenary_sentence_str}\n")
                file.write(f"Plenary tokens: {plenary_tokens_str}\n")
                file.write(f"Probability of plenary sentence in plenary corpus: {round(plenary_prob_in_plenary, 2)}\n")
                file.write(f"Probability of plenary sentence in committee corpus: {round(plenary_prob_in_committee, 2)}\n")

                # determine likely corpus
                if committee_prob_in_committee > committee_prob_in_plenary:
                    likely_corpus = "committee"
                else:
                    likely_corpus = "plenary"
                file.write(f"This sentence is more likely to appear in corpus: {likely_corpus}\n\n")
    except Exception as error:
        print(f"Error occurred while writing to TXT '{file_path}': {error}")

def read_txt_to_matrix(file_name):
    """
    reads a text file and converts it into a matrix of words
    """
    try:
        matrix_txt = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line_txt in file:
                row = [x for x in line_txt.strip().split()]
                matrix_txt.append(row)
        return matrix_txt
    except Exception as error:
        print(f"Error occurred while reading from TXT '{file_name}': {error}")
        return []

if __name__ == "__main__":
    # read data from 'result.jsonl'
    matrix_c, matrix_p = read_jsonl('result.jsonl')

    # initialize language models
    c_lm = Trigram_LM(matrix_c)
    p_lm = Trigram_LM(matrix_p)

    # extract collocations
    lists_c_p = {
        'Two-gram': {
            'Committee': c_lm.get_k_n_collocations(10, 2),
            'Plenary': p_lm.get_k_n_collocations(10, 2)
        },
        'Three-gram': {
            'Committee': c_lm.get_k_n_collocations(10, 3),
            'Plenary': p_lm.get_k_n_collocations(10, 3)
        },
        'Four-gram': {
            'Committee': c_lm.get_k_n_collocations(10, 4),
            'Plenary': p_lm.get_k_n_collocations(10, 4)
        }
    }
    write_txt_collocations('knesset_collocations.txt', lists_c_p)

    # select 4 sentences for manual iteration
    # as per the latest user instruction
    # selecting 4 sentences, masking 10%
    sampled_sents = random.sample([s for s in matrix_c if len(s) >= 5], 4)
    masked_sents = mask_tokens_in_sentences(sampled_sents, 10)

    # write sampled and masked sentences to a temporary file if needed
    # but according to user instruction, they want to manually choose best weights based on predictions each epoch
    # so we proceed to iterate

    # set comma_penalty as a global variable (can be manually updated)
    comma_penalty = 0.5  # you can update this value manually

    # iterate over 50 epochs
    for epoch in range(50):
        # randomly choose lambdas for interpolation that sum to 1
        a, b, c = random.random(), random.random(), random.random()
        total = a + b + c
        l1, l2, l3 = a / total, b / total, c / total

        # rebuild models with current lambdas
        c_lm_epoch = Trigram_LM(matrix_c)
        p_lm_epoch = Trigram_LM(matrix_p)

        # update lambdas in models
        c_lm_epoch.l1, c_lm_epoch.l2, c_lm_epoch.l3 = l1, l2, l3
        p_lm_epoch.l1, p_lm_epoch.l2, p_lm_epoch.l3 = l1, l2, l3

        print(f"epoch {epoch+1}: lambdas=({l1:.3f}, {l2:.3f}, {l3:.3f}), comma_penalty={comma_penalty:.3f}")

        # predict and display predictions for each masked sentence
        for idx, (orig, masked) in enumerate(zip(sampled_sents, masked_sents)):
            om_toks = masked.split()
            fi_toks = om_toks.copy()
            pred_tokens = []
            for i, tk in enumerate(fi_toks):
                if tk == '[*]':
                    context = ' '.join(fi_toks[:i])
                    next_tk = p_lm_epoch.generate_next_token(context, comma_penalty)
                    fi_toks[i] = next_tk
                    pred_tokens.append(next_tk)
            filled_sent = ' '.join(fi_toks)

            print(f"  sentence {idx+1}:")
            print(f"    original: {' '.join(orig)}")
            print(f"    masked: {' '.join(om_toks)}")
            print(f"    filled: {filled_sent}")
            print(f"    predicted tokens: {', '.join(pred_tokens)}\n")

        # user can manually update `comma_penalty` here based on the predictions
        # for example, after observing epoch results, adjust `comma_penalty` accordingly
        # since the user wants to manually inspect, we do not automate the update

    # Note:
    # - The script selects 4 sentences, masks 10% of their tokens, and runs 50 epochs.
    # - In each epoch, it randomly selects lambdas for interpolation, rebuilds the language models,
    #   predicts the masked tokens with the current `comma_penalty`, and prints the predictions.
    # - The user can observe the printed predictions and manually adjust the `comma_penalty` variable in the code
    #   to improve prediction accuracy by penalizing commas as needed.
