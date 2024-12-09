import collections
import csv
import math


def read_csv(file_path):
    try:
        # open the CSV file
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            # create a csv reader
            csv_reader = csv.DictReader(csv_file)
            word_matrix_c = []
            word_matrix_p = []
            # go through each row in the csv file
            for row in csv_reader:
                # get the sentence_text from the row
                if row.get('sentence_text'):
                    sentence = row['sentence_text']

                    # split the sentence into words
                    words = sentence.split()
                    temp = []
                    for word in words:
                        temp.append(word)
                    # add the clean words to the right matrix
                    if row.get('protocol_type') == 'plenary':
                        word_matrix_p.append(temp)
                    else:
                        word_matrix_c.append(temp)
                # get the sentence_text from the row
                elif row.get('text_sentence'):
                    sentence = row['text_sentence']
                    # split the sentence into words
                    words = sentence.split()
                    temp = []
                    for word in words:
                        temp.append(word)
                    # add the clean words to the right matrix
                    if row.get('type_protocol') == 'plenary':
                        word_matrix_p.append(temp)
                    else:
                        word_matrix_c.append(temp)
            return word_matrix_c, word_matrix_p
    except Exception as error:
        print("Error occurred while reading from CSV '" + file_path + "': " + str(error))


class Trigram_LM:

    def __init__(self, word_matrix):
        # save the words of the corpus
        self.word_matrix = word_matrix
        # save the amount of 3 words to appear together
        self.trigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        # save the amount of 2 words to appear together
        self.bigram_counts = collections.defaultdict(int)
        # save the amount of a word to appear
        self.unigram_counts = collections.defaultdict(int)
        self.N = 0
        self.V = 0
        self.analyze_matrix()

    def analyze_matrix(self):
        mat = self.word_matrix
        # the word matrix save all the words of a sentence together, so we flatten the matrix.
        mat = [word for sentence in mat for word in sentence]
        # save the number of words in the corpus
        self.N = len(mat)
        # calculate the 3-gram, 2-gram and 1-gram
        for i in range(2, len(mat)):
            w1, w2, w3 = mat[i - 2], mat[i - 1], mat[i]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.bigram_counts[(w1, w2)] += 1
            self.unigram_counts[w3] += 1
        # save the number of different words in the corpus
        self.V = len(self.unigram_counts)

    def calculate_prob_of_sentence(self, tokens, smoothing,
                                   lambdas=[8999999 / 10000000, 1000000 / 10000000, 1 / 10000000]):
        # split the string into tokens
        tokens = tokens.split()
        sol = 0

        # if no words in corpus or no words in sentence return -inf
        if self.V == 0 or len(tokens) == 0:
            return -math.inf

        # if the number of tokens is 1 calculate it as an unigram
        elif len(tokens) == 1:
            sol = math.log2((self.unigram_counts[tokens[0]] + 1) / (self.N + self.V))

        # if the number of tokens is 2 calculate it as a 2-gram
        elif len(tokens) == 2:
            bigram_count = self.bigram_counts[(tokens[0], tokens[1])]
            if smoothing == 'Laplace':
                sol = math.log2((bigram_count + 1) / (self.unigram_counts[tokens[0]] + self.V))
            else:
                sol = math.log2(lambdas[0] * (bigram_count + 1) / (self.unigram_counts[tokens[0]] + self.V) + \
                                (lambdas[1] * (self.unigram_counts[tokens[1]] + 1) / (self.N + self.V)))

        # if the number of tokens is bigger than 2 calculate it as an 3-gram
        else:
            log = 0
            # if we get a number larger than 3
            for i in range(2, len(tokens)):
                # we that the 3 current words
                w1, w2, w3 = tokens[i - 2], tokens[i - 1], tokens[i]
                # calculate the appearance of the 3 words together
                trigram_count = self.trigram_counts[(w1, w2)][w3]
                # calculate the appearance of the first 2 words together
                bigram_count = self.bigram_counts[(w1, w2)]

                if smoothing == 'Laplace':
                    # the formula as we learned in class for laplace smoothing
                    smoothed_probability = (trigram_count + 1) / (bigram_count + self.V)

                else:
                    # Calculate probabilities
                    trigram_prob = (trigram_count + 1) / (bigram_count + self.V)
                    bigram_prob = (self.bigram_counts[(w2, w3)] + 1) / (self.unigram_counts[w2] + self.V)
                    unigram_prob = (self.unigram_counts[w3] + 1) / (self.N + self.V)
                    # the formula as we learned in class for linear interpolation
                    smoothed_probability = lambdas[0] * trigram_prob + lambdas[1] * bigram_prob + lambdas[
                        2] * unigram_prob
                # sum the log probability for every 3 consecutive words
                log += math.log2(smoothed_probability)
            sol = log
        print(round(sol, 3))
        return sol

    def generate_next_token(self, tokens, smoothing='Laplace', lambdas=[8999999 / 10000000, 1000000 / 10000000, 1 / 10000000]):
        # split the string into tokens
        tokens = tokens.split()
        sentence = ''
        # check if we have more than 3 words
        if len(tokens) >= 2:
            # take the first word of the 3 last words
            w1 = tokens[-2]
            # put the first word in the sentence
            sentence += w1 + ' '
        # check if we have more than 2 words
        if len(tokens) >= 1:
            # take the second word of the 3 last words
            w2 = tokens[-1]
            # put the second word in the sentence
            sentence += w2 + ' '
        # the max probability
        maxi = -math.inf
        # the word with the maximum probability
        word = ''
        # go over all the different words in the corpus to find the word with the max probability
        for w3 in list(self.unigram_counts.keys()):
            # calculate the probability of the current word to be the next in the sentence
            prob = self.calculate_prob_of_sentence(sentence + w3, smoothing, lambdas)
            # check if the probability is bigger than the current maximum
            if maxi <= prob:
                # if not update the parameters
                maxi = prob
                word = w3
        print(word)
        return word

    def get_k_n_collocations(self, k, n):
        # a dictionary for contain the collocation and its pmi
        collocations = {}
        # go over all the sentences in the corpus
        for sentence in self.word_matrix:
            # go over all the sentences parts with length n
            for index in range(n, len(sentence)):
                pmi = 0
                # get the n current words
                words = sentence[index - n: index]
                # we will calculate the pmi only for collocations we didn't see
                if tuple(words) not in collocations:
                    # if the collocation is of len 2
                    if len(words) == 2:
                        # the formula as we saw in class
                        bigram_prob = self.bigram_counts[(words[0], words[1])] / self.unigram_counts[words[0]]
                        unigram_w1_prob = self.unigram_counts[words[0]] / self.N
                        unigram_w2_prob = self.unigram_counts[words[1]] / self.N
                        pmi = math.log2(bigram_prob / (unigram_w2_prob * unigram_w1_prob))
                    else:
                        # if the collocation is of len 3 or higher
                        for i in range(2, len(words)):
                            # get the 3 current words
                            w1 = words[i - 2]
                            w2 = words[i - 1]
                            w3 = words[i]
                            # the formula as we saw in the forum
                            trigram_prob = self.trigram_counts[(w1, w2)][w3] / self.bigram_counts[(w1, w2)]
                            unigram_w1_prob = self.unigram_counts[w1] / self.N
                            unigram_w2_prob = self.unigram_counts[w2] / self.N
                            unigram_w3_prob = self.unigram_counts[w3] / self.N
                            # sum the pmi for every 3 consecutive words
                            pmi += math.log2(trigram_prob / (unigram_w1_prob * unigram_w2_prob * unigram_w3_prob))
                    # save the collocation and its pmi
                    collocations[tuple(words)] = pmi
        # sort by pmi
        collocations = sorted(collocations.items(), key=lambda x: x[1], reverse=True)
        # return only the k most common collocation
        return collocations[:k]

    def fill_tokens(self, sentence):
        # save the last words up to 2
        partial_sentence = []
        # help as convert the list to a string
        convert = " "
        # save the words we change from [*]
        changed_words = []
        # go over the sentence
        for i, word in enumerate(sentence):
            # got to an empty word
            if word == '[*]':
                # we generate the word by the last 2 words we saw
                word = self.generate_next_token(convert.join(partial_sentence), 'Linear')
                # save the word we generated in the place of [*]
                sentence[i] = word
                # add the chosen word to list
                changed_words.append(word)
            # update the partial sentence list
            # if we have 2 words
            if len(partial_sentence) > 1:
                # push the second word to be the first
                partial_sentence[0] = partial_sentence[1]
                # put the new word to be second
                partial_sentence[1] = word
            # if we have less than 2 words
            else:
                # add the word
                partial_sentence.append(word)

        return changed_words, sentence


def write_txt_collocations(file_path, lists):
    try:
        # open the CSV file
        with open(file_path, 'w', encoding='utf-8') as file:
            # go over the different n-grams (2,3,4)
            for collocation_type, corpus_collocations in lists.items():
                # write the n-gram number
                file.write(f"{collocation_type} collocations:\n")
                # go over the corpus type (c/p)
                for corpus, collocation_list in corpus_collocations.items():
                    # write the corpus type
                    file.write(f"{corpus} corpus:\n")
                    # go over the collocations
                    for collocation_tuple in collocation_list:
                        collocation, pmi_value = collocation_tuple
                        # turn to string and add space between words
                        formatted_collocation = ' '.join(collocation)
                        # write the info to the csv file
                        file.write(formatted_collocation + '\n')
                    # add new line between paragraphs
                    file.write('\n')
    except Exception as error:
        print("Error occurred while writing to CSV '" + file_path + "': " + str(error))


def write_generated_sentences_to_txt(file_path, original_sentences, committee_lm, plenary_lm):
    try:
        # open the CSV file
        with open(file_path, 'w', encoding='utf-8') as file:
            # go over the sentences
            for original_sentence in original_sentences:
                # write the sentence
                file.write(f"Original sentence: {' '.join(original_sentence)}\n")

                # use fill_tokens to turn [*] to words by the committee model
                committee_sentence = committee_lm.fill_tokens(original_sentence.copy())
                # turn the list to string and divide the words with space
                committee_sentence_str = ' '.join(committee_sentence[1])
                # turn the changed tokens to string and divided with ','
                committee_tokens_str = ', '.join(committee_sentence[0])
                # calculate the probability of the sentence by the committee model
                committee_prob_in_committee_corpus = committee_lm.calculate_prob_of_sentence(committee_sentence_str,
                                                                                             'Laplace')
                # calculate the probability of the sentence by the plenary model
                committee_prob_in_plenary_corpus = plenary_lm.calculate_prob_of_sentence(committee_sentence_str, 'Laplace')

                # write the info
                file.write(f"Committee sentence: {committee_sentence_str}\n")
                file.write(f"Committee tokens: {committee_tokens_str}\n")
                file.write(f"Probability of committee sentence in committee corpus: {round(committee_prob_in_committee_corpus, 3)}\n")
                file.write(f"Probability of committee sentence in plenary corpus: {round(committee_prob_in_plenary_corpus, 3)}\n")

                # use fill_tokens to turn [*] to words by the plenary model
                plenary_sentence = plenary_lm.fill_tokens(original_sentence.copy())
                # turn the list to string and divide the words with space
                plenary_sentence_str = ' '.join(plenary_sentence[1])
                # turn the changed tokens to string and divided with ','
                plenary_tokens_str = ', '.join(plenary_sentence[0])
                # calculate the probability of the sentence by the plenary model
                plenary_prob_in_plenary_corpus = plenary_lm.calculate_prob_of_sentence(plenary_sentence_str, 'Laplace')
                # calculate the probability of the sentence by the committee model
                plenary_prob_in_committee_corpus = committee_lm.calculate_prob_of_sentence(plenary_sentence_str, 'Laplace')

                # write info
                file.write(f"Plenary sentence: {plenary_sentence_str}\n")
                file.write(f"Plenary tokens: {plenary_tokens_str}\n")
                file.write(f"Probability of plenary sentence in plenary corpus: {round(plenary_prob_in_plenary_corpus, 3)}\n")
                file.write(f"Probability of plenary sentence in committee corpus: {round(plenary_prob_in_committee_corpus, 3)}\n")

                # check if the sentence is more likely to be in committee corpus or plenary corpus
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
        # open the CSV file
        with open(file_name, 'r', encoding='utf-8') as file:
            # read evey line and add to matrix
            for line_txt in file:
                row = [x for x in line_txt.strip().split()]
                matrix_txt.append(row)
        return matrix_txt
    except Exception as error:
        print("Error occurred while reading from CSV '" + file_name + "': " + str(error))


if __name__ == "__main__":
    # we read the csv and divided the info to committee and plenary
    matrix_c, matrix_p = read_csv('example_knesset_corpus.csv')
    # part 1 - created the language model
    c_lm = Trigram_LM(matrix_c)
    p_lm = Trigram_LM(matrix_p)

    # part 2 - create data structure of the 2, 3, 4 most common collocation
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
    # write the file of the second part
    write_txt_collocations('knesset_collocations.txt', lists_c_p)

    # part 3 - read the file and create matrix for the sentences
    matrix = read_txt_to_matrix('masked_sentences.txt')
    origin_sentences = []
    for line in matrix:
        origin_sentences.append(line)
    # write the file of the third part
    write_generated_sentences_to_txt('sentences_results.txt', origin_sentences, c_lm, p_lm)

