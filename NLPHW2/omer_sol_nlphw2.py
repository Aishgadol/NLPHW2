import collections
import json
import math

def read_jsonl(file_path):
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
                # We assume `type_protocol` is either "committee" or "plenary"
                # and `text_sentence` is the tokenized sentence.
                protocol_type = obj.get('type_protocol', '').strip().lower()
                sentence = obj.get('text_sentence', '').strip()
                if sentence and protocol_type in ['committee', 'plenary']:
                    words = sentence.split()
                    if protocol_type == 'plenary':
                        word_matrix_p.append(words)
                    else:
                        word_matrix_c.append(words)
        return word_matrix_c, word_matrix_p
    except Exception as error:
        print("Error occurred while reading from JSONL '" + file_path + "': " + str(error))

class Trigram_LM:
    def __init__(self, word_matrix):
        self.word_matrix = word_matrix
        self.trigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.bigram_counts = collections.defaultdict(int)
        self.unigram_counts = collections.defaultdict(int)
        self.N = 0
        self.V = 0
        self.analyze_matrix()

    def analyze_matrix(self):
        mat = self.word_matrix
        mat = [word for sentence in mat for word in sentence]
        self.N = len(mat)
        for i in range(2, len(mat)):
            w1, w2, w3 = mat[i - 2], mat[i - 1], mat[i]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.bigram_counts[(w1, w2)] += 1
            self.unigram_counts[w3] += 1
        self.V = len(self.unigram_counts)

    def calculate_prob_of_sentence(self, tokens, smoothing, lambdas=[8999999/10000000, 1000000/10000000, 1/10000000]):
        tokens = tokens.split()
        sol = 0
        if self.V == 0 or len(tokens) == 0:
            return -math.inf
        elif len(tokens) == 1:
            sol = math.log2((self.unigram_counts[tokens[0]] + 1) / (self.N + self.V))
        elif len(tokens) == 2:
            bigram_count = self.bigram_counts[(tokens[0], tokens[1])]
            if smoothing == 'Laplace':
                sol = math.log2((bigram_count + 1) / (self.unigram_counts[tokens[0]] + self.V))
            else:
                sol = math.log2(lambdas[0] * (bigram_count + 1) / (self.unigram_counts[tokens[0]] + self.V) +
                                (lambdas[1] * (self.unigram_counts[tokens[1]] + 1) / (self.N + self.V)))
        else:
            log_val = 0
            for i in range(2, len(tokens)):
                w1, w2, w3 = tokens[i - 2], tokens[i - 1], tokens[i]
                trigram_count = self.trigram_counts[(w1, w2)][w3]
                bigram_count = self.bigram_counts[(w1, w2)]
                if smoothing == 'Laplace':
                    smoothed_probability = (trigram_count + 1) / (bigram_count + self.V)
                else:
                    trigram_prob = (trigram_count + 1) / (bigram_count + self.V)
                    bigram_prob = (self.bigram_counts[(w2, w3)] + 1) / (self.unigram_counts[w2] + self.V)
                    unigram_prob = (self.unigram_counts[w3] + 1) / (self.N + self.V)
                    smoothed_probability = lambdas[0] * trigram_prob + lambdas[1] * bigram_prob + lambdas[2] * unigram_prob
                log_val += math.log2(smoothed_probability)
            sol = log_val
        print(round(sol, 3))
        return sol

    def generate_next_token(self, tokens, smoothing='Laplace', lambdas=[8999999/10000000, 1000000/10000000, 1/10000000]):
        tokens = tokens.split()
        sentence = ''
        if len(tokens) >= 2:
            w1 = tokens[-2]
            sentence += w1 + ' '
        if len(tokens) >= 1:
            w2 = tokens[-1]
            sentence += w2 + ' '
        maxi = -math.inf
        word = ''
        for w3 in list(self.unigram_counts.keys()):
            prob = self.calculate_prob_of_sentence(sentence + w3, smoothing, lambdas)
            if maxi <= prob:
                maxi = prob
                word = w3
        print(word)
        return word

    def get_k_n_collocations(self, k, n):
        collocations = {}
        for sentence in self.word_matrix:
            for index in range(n, len(sentence)):
                pmi = 0
                words = sentence[index - n: index]
                if tuple(words) not in collocations:
                    if len(words) == 2:
                        bigram_prob = self.bigram_counts[(words[0], words[1])] / self.unigram_counts[words[0]] if self.unigram_counts[words[0]] != 0 else 0
                        unigram_w1_prob = self.unigram_counts[words[0]] / self.N if self.N != 0 else 0
                        unigram_w2_prob = self.unigram_counts[words[1]] / self.N if self.N != 0 else 0
                        if unigram_w1_prob != 0 and unigram_w2_prob != 0:
                            pmi = math.log2(bigram_prob / (unigram_w2_prob * unigram_w1_prob)) if (unigram_w2_prob * unigram_w1_prob) > 0 else 0
                    else:
                        for i in range(2, len(words)):
                            w1 = words[i - 2]
                            w2 = words[i - 1]
                            w3 = words[i]
                            if self.bigram_counts[(w1, w2)] != 0:
                                trigram_prob = self.trigram_counts[(w1, w2)][w3] / self.bigram_counts[(w1, w2)]
                            else:
                                trigram_prob = 0
                            unigram_w1_prob = self.unigram_counts[w1] / self.N if self.N != 0 else 0
                            unigram_w2_prob = self.unigram_counts[w2] / self.N if self.N != 0 else 0
                            unigram_w3_prob = self.unigram_counts[w3] / self.N if self.N != 0 else 0
                            denom = (unigram_w1_prob * unigram_w2_prob * unigram_w3_prob)
                            if denom > 0:
                                pmi += math.log2(trigram_prob / denom) if trigram_prob > 0 else 0
                    collocations[tuple(words)] = pmi
        collocations = sorted(collocations.items(), key=lambda x: x[1], reverse=True)
        return collocations[:k]

    def fill_tokens(self, sentence):
        partial_sentence = []
        convert = " "
        changed_words = []
        for i, word in enumerate(sentence):
            if word == '[*]':
                word = self.generate_next_token(convert.join(partial_sentence), 'Linear')
                sentence[i] = word
                changed_words.append(word)
            if len(partial_sentence) > 1:
                partial_sentence[0] = partial_sentence[1]
                partial_sentence[1] = word
            else:
                partial_sentence.append(word)
        return changed_words, sentence

def write_txt_collocations(file_path, lists):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for collocation_type, corpus_collocations in lists.items():
                file.write(f"{collocation_type} collocations:\n")
                for corpus, collocation_list in corpus_collocations.items():
                    file.write(f"{corpus} corpus:\n")
                    for collocation_tuple in collocation_list:
                        collocation, pmi_value = collocation_tuple
                        formatted_collocation = ' '.join(collocation)
                        file.write(formatted_collocation + '\n')
                    file.write('\n')
    except Exception as error:
        print("Error occurred while writing to CSV '" + file_path + "': " + str(error))

def write_generated_sentences_to_txt(file_path, original_sentences, committee_lm, plenary_lm):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for original_sentence in original_sentences:
                file.write(f"Original sentence: {' '.join(original_sentence)}\n")

                committee_sentence = committee_lm.fill_tokens(original_sentence.copy())
                committee_sentence_str = ' '.join(committee_sentence[1])
                committee_tokens_str = ', '.join(committee_sentence[0])
                committee_prob_in_committee_corpus = committee_lm.calculate_prob_of_sentence(committee_sentence_str, 'Laplace')
                committee_prob_in_plenary_corpus = plenary_lm.calculate_prob_of_sentence(committee_sentence_str, 'Laplace')

                file.write(f"Committee sentence: {committee_sentence_str}\n")
                file.write(f"Committee tokens: {committee_tokens_str}\n")
                file.write(f"Probability of committee sentence in committee corpus: {round(committee_prob_in_committee_corpus, 3)}\n")
                file.write(f"Probability of committee sentence in plenary corpus: {round(committee_prob_in_plenary_corpus, 3)}\n")

                plenary_sentence = plenary_lm.fill_tokens(original_sentence.copy())
                plenary_sentence_str = ' '.join(plenary_sentence[1])
                plenary_tokens_str = ', '.join(plenary_sentence[0])
                plenary_prob_in_plenary_corpus = plenary_lm.calculate_prob_of_sentence(plenary_sentence_str, 'Laplace')
                plenary_prob_in_committee_corpus = committee_lm.calculate_prob_of_sentence(plenary_sentence_str, 'Laplace')

                file.write(f"Plenary sentence: {plenary_sentence_str}\n")
                file.write(f"Plenary tokens: {plenary_tokens_str}\n")
                file.write(f"Probability of plenary sentence in plenary corpus: {round(plenary_prob_in_plenary_corpus, 3)}\n")
                file.write(f"Probability of plenary sentence in committee corpus: {round(plenary_prob_in_committee_corpus, 3)}\n")

                if committee_prob_in_committee_corpus > committee_prob_in_plenary_corpus:
                    likely_corpus = "committee"
                else:
                    likely_corpus = "plenary"
                file.write(f"This sentence is more likely to appear in corpus: {likely_corpus}\n\n")
    except Exception as error:
        print("Error occurred while writing to CSV '" + file_path + "': " + str(error))

def read_txt_to_matrix(file_name):
    try:
        matrix_txt = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line_txt in file:
                row = [x for x in line_txt.strip().split()]
                matrix_txt.append(row)
        return matrix_txt
    except Exception as error:
        print("Error occurred while reading from CSV '" + file_name + "': " + str(error))

if __name__ == "__main__":
    # Instead of CSV, we now read from a JSONL file.
    # This assumes the JSONL file has `type_protocol` ("committee"/"plenary") and `text_sentence`.
    matrix_c, matrix_p = read_jsonl('result.jsonl')

    c_lm = Trigram_LM(matrix_c)
    p_lm = Trigram_LM(matrix_p)

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

    matrix = read_txt_to_matrix('masked_sentences.txt')
    origin_sentences = []
    for line in matrix:
        origin_sentences.append(line)
    write_generated_sentences_to_txt('sentences_results.txt', origin_sentences, c_lm, p_lm)
