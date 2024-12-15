# # 12/12/24, 18:58, best code right now:
# # import json
# # import math
# # import random
# # import os
# # import sys
# # import collections
# # import pandas as pd
# # import re
# #
# # # this code builds trigram-based language models with linear interpolation and laplace smoothing
# # # we now have updated conditions:
# # # 1. column names have changed: 'type_protocol' -> 'protocol_type', 'text_sentence' -> 'sentence_text'
# # # 2. we can now predict punctuation if its likelihood is truly the highest
# # # 3. we can tune constants (weights for interpolation and so on) to improve perplexity and predictions
# # # 4. we must consider all n-grams (unigram, bigram, trigram) carefully for best predictions, and we can rely on
# # #    any allowed technique (only interpolation and laplace smoothing are allowed by the original instructions)
# # # 5. we add comments in lowercase explaining what each method/line does
# #
# # # assumptions:
# # # - we rely on linear interpolation of unigram, bigram, trigram probabilities with laplace smoothing
# # # - we pick constants for interpolation after experimenting: we choose (0.05, 0.15, 0.8), giving strong weight to trigram
# # # - if punctuation has higher probability than all other tokens, we allow punctuation
# # # - we only use known tokens from the training corpus
# # # - for collocations and tf-idf, we keep the methods from before
# # # - we produce the same output files as required
# # # - we mask 10% of tokens and ensure each sampled sentence has at least 5 tokens
# #
# # hebrew_re = re.compile(r'[\u0590-\u05FF]+')
# #
# # def tokenize_sentence(sent):
# #     # splits a sentence into tokens by whitespace
# #     return sent.strip().split()
# #
# # def is_hebrew_word(w):
# #     # checks if a token is purely a hebrew word
# #     return bool(hebrew_re.fullmatch(w))
# #
# # class LM_Trigram:
# #     def __init__(self, sentences, lambdas=(0.05,0.15,0.8)):
# #         # builds trigram model with given lambdas for interpolation
# #         # uses laplace smoothing
# #         # sentences is a list of list of tokens
# #         self.l1, self.l2, self.l3 = lambdas
# #         self.unigram_counts = collections.Counter()
# #         self.bigram_counts = collections.Counter()
# #         self.trigram_counts = collections.Counter()
# #
# #         # count n-grams
# #         for sent in sentences:
# #             s = ["_s0_", "_s1_"] + sent
# #             for w in s:
# #                 self.unigram_counts[w]+=1
# #             for i in range(len(s)-1):
# #                 self.bigram_counts[(s[i], s[i+1])]+=1
# #             for i in range(len(s)-2):
# #                 self.trigram_counts[(s[i], s[i+1], s[i+2])]+=1
# #
# #         self.v = len(self.unigram_counts)
# #         self.total_unigrams = sum(self.unigram_counts.values())
# #
# #     def _laplace_unigram(self, w):
# #         # laplace smoothing for unigram
# #         return (self.unigram_counts[w] + 1)/(self.total_unigrams + self.v)
# #
# #     def _laplace_bigram(self, w1, w2):
# #         # laplace smoothing for bigram
# #         count_bi = self.bigram_counts[(w1,w2)]
# #         count_uni = self.unigram_counts[w1]
# #         return (count_bi + 1)/(count_uni + self.v)
# #
# #     def _laplace_trigram(self, w1, w2, w3):
# #         # laplace smoothing for trigram
# #         count_tri = self.trigram_counts[(w1,w2,w3)]
# #         count_bi = self.bigram_counts[(w1,w2)]
# #         return (count_tri + 1)/(count_bi + self.v)
# #
# #     def _interp_prob(self, w1, w2, w3):
# #         # computes interpolated probability using chosen lambdas
# #         p_uni = self._laplace_unigram(w3)
# #         p_bi = self._laplace_bigram(w2, w3)
# #         p_tri = self._laplace_trigram(w1,w2,w3)
# #         return self.l1*p_uni + self.l2*p_bi + self.l3*p_tri
# #
# #     def calculate_prob_of_sentence(self, sentence):
# #         # calculates log probability of a sentence using the trigram model
# #         tokens = sentence.strip().split()
# #         tokens = ["_s0_", "_s1_"] + tokens
# #         log_prob=0.0
# #         for i in range(2, len(tokens)):
# #             p=self._interp_prob(tokens[i-2], tokens[i-1], tokens[i])
# #             log_prob+=math.log(p)
# #         return log_prob
# #
# #     def generate_next_token(self, context):
# #         # generates the next token based on highest probability
# #         # now we allow punctuation if it truly is the highest probability
# #         # we do not exclude punctuation or non-hebrew words now; user said we can predict punctuation if surpasses others
# #         # but we must produce the best perplexity: let's just pick the highest probability token from entire vocab
# #         ctx = context.strip().split()
# #         if len(ctx)<2:
# #             ctx = ["_s0_","_s1_"] + ctx
# #         w1,w2=ctx[-2], ctx[-1]
# #         max_token=None
# #         max_prob=-1
# #         # check all tokens in vocabulary
# #         # we do not restrict to hebrew only now, since user says we can predict punctuation if higher
# #         # if we want best perplexity and predictions, we trust the model probabilities
# #         # to handle the request from previous steps about punctuation was changed now;
# #         # user now says we can predict punctuation if surpasses everything else.
# #         # so we consider all tokens:
# #         for w3 in self.unigram_counts:
# #             p=self._interp_prob(w1,w2,w3)
# #             if p>max_prob:
# #                 max_prob=p
# #                 max_token=w3
# #         return (max_token, math.log(max_prob))
# #
# # def get_k_n_t_collocations(k, n, t, corpus_df, metric_type="frequency"):
# #     # extracts top k collocations of length n with at least t occurrences
# #     # supports frequency and tf-idf metrics
# #     docs = corpus_df["sentence_text"].tolist()
# #     ngram_doc_freq = {}
# #     for i,doc in enumerate(docs):
# #         tokens = tokenize_sentence(doc)
# #         ngs = zip(*[tokens[j:] for j in range(n)])
# #         doc_count = collections.Counter(ngs)
# #         for ng,c in doc_count.items():
# #             if ng not in ngram_doc_freq:
# #                 ngram_doc_freq[ng]={}
# #             ngram_doc_freq[ng][i]=c
# #
# #     global_count = {ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
# #     filtered = {ng: freq for ng,freq in global_count.items() if freq>=t}
# #
# #     if metric_type=="frequency":
# #         sorted_ngrams=sorted(filtered.items(), key=lambda x:x[1], reverse=True)
# #         return [(" ".join(ng), val) for ng,val in sorted_ngrams[:k]]
# #     elif metric_type=="tfidf":
# #         D=len(docs)
# #         doc_appearances={ng: len(ngram_doc_freq[ng]) for ng in filtered}
# #         scores={}
# #         for ng in filtered:
# #             df_count=doc_appearances[ng]
# #             if df_count==0:
# #                 continue
# #             idf = math.log(D/df_count)
# #             total_score=0.0
# #             for d_i,freq_ng in ngram_doc_freq[ng].items():
# #                 tokens_count=len(tokenize_sentence(docs[d_i]))
# #                 tf = freq_ng/tokens_count if tokens_count>0 else 0
# #                 total_score+=tf*idf
# #             scores[ng]=total_score
# #         sorted_scores=sorted(scores.items(), key=lambda x:x[1], reverse=True)
# #         return [(" ".join(ng), sc) for ng,sc in sorted_scores[:k]]
# #     else:
# #         return []
# #
# # def mask_tokens_in_sentences(sentences, x):
# #     # masks x% of tokens in each sentence
# #     masked=[]
# #     for s in sentences:
# #         toks=s.strip().split()
# #         if len(toks)==0:
# #             masked.append(s)
# #             continue
# #         num_to_mask=max(1,int(len(toks)*x/100))
# #         if num_to_mask>len(toks):
# #             num_to_mask=len(toks)
# #         idxs=random.sample(range(len(toks)), num_to_mask)
# #         for i in idxs:
# #             toks[i]="[*]"
# #         masked.append(" ".join(toks))
# #     return masked
# #
# # def main():
# #     # reading input
# #     corpus_file=sys.argv[1]
# #     output_dir=sys.argv[2]
# #
# #     data=[]
# #     # column names changed: now we have 'protocol_type' and 'sentence_text'
# #     # we assume json has 'protocol_type' and 'sentence_text' fields now
# #     with open(corpus_file,'r',encoding='utf-8') as f:
# #         for line in f:
# #             line=line.strip()
# #             if not line:
# #                 continue
# #             obj=json.loads(line)
# #             t=obj['protocol_type'].strip().lower()
# #             txt=obj['sentence_text'].strip()
# #             data.append((t,txt))
# #     df=pd.DataFrame(data,columns=["protocol_type","sentence_text"])
# #
# #     committee_df = df[df.protocol_type=="committee"].copy()
# #     plenary_df = df[df.protocol_type=="plenary"].copy()
# #
# #     committee_sents=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
# #     plenary_sents=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]
# #
# #     # we can try different lambdas, after experimentation we pick (0.05,0.15,0.8)
# #     # this heavily favors trigram probability
# #     # we do nothing special besides normal laplace smoothing
# #     committee_lm=LM_Trigram(committee_sents, lambdas=(0.05,0.15,0.8))
# #     plenary_lm=LM_Trigram(plenary_sents, lambdas=(0.05,0.15,0.8))
# #
# #     with open(os.path.join(output_dir,"txt.collocations_knesset"),"w",encoding="utf-8") as f:
# #         for n in [2,3,4]:
# #             if n==2:
# #                 f.write("Two-gram collocations:\n")
# #             elif n==3:
# #                 f.write("Three-gram collocations:\n")
# #             else:
# #                 f.write("Four-gram collocations:\n")
# #
# #             f.write("Frequency:\nCommittee corpus:\n")
# #             freq_comm = get_k_n_t_collocations(10,n,5,committee_df,"frequency")
# #             for c_val in freq_comm:
# #                 f.write(c_val[0]+"\n")
# #             f.write("\nPlenary corpus:\n")
# #             freq_plen = get_k_n_t_collocations(10,n,5,plenary_df,"frequency")
# #             for p_val in freq_plen:
# #                 f.write(p_val[0]+"\n")
# #             f.write("\n")
# #
# #             f.write("TF-IDF:\nCommittee corpus:\n")
# #             tfidf_comm = get_k_n_t_collocations(10,n,5,committee_df,"tfidf")
# #             for c_val in tfidf_comm:
# #                 f.write(c_val[0]+"\n")
# #             f.write("\nPlenary corpus:\n")
# #             tfidf_plen = get_k_n_t_collocations(10,n,5,plenary_df,"tfidf")
# #             for p_val in tfidf_plen:
# #                 f.write(p_val[0]+"\n")
# #             f.write("\n")
# #
# #     committee_texts = committee_df["sentence_text"].tolist()
# #     candidate_sents=[s for s in committee_texts if len(tokenize_sentence(s))>=5]
# #     random.seed(42)
# #     sampled_sents = random.sample(candidate_sents, 10)
# #
# #     masked_sents = mask_tokens_in_sentences(sampled_sents,10)
# #
# #     with open(os.path.join(output_dir,"txt.sents_sampled_original"),"w",encoding="utf-8") as f:
# #         for s in sampled_sents:
# #             f.write(s+"\n")
# #
# #     with open(os.path.join(output_dir,"txt.sents_sampled_masked"),"w",encoding="utf-8") as f:
# #         for s in masked_sents:
# #             f.write(s+"\n")
# #
# #     results=[]
# #     # predict masked tokens using plenary model
# #     for orig,masked in zip(sampled_sents,masked_sents):
# #         toks_masked=masked.strip().split()
# #         pred_tokens=[]
# #         for i,tk in enumerate(toks_masked):
# #             if tk=="[*]":
# #                 ctx=" ".join(toks_masked[:i])
# #                 # generate token with punctuation allowed if it's highest
# #                 next_tk, next_log_p=plenary_lm.generate_next_token(ctx)
# #                 toks_masked[i]=next_tk
# #                 pred_tokens.append(next_tk)
# #         filled_sent=" ".join(toks_masked)
# #
# #         pp_plen=plenary_lm.calculate_prob_of_sentence(filled_sent)
# #         pp_comm=committee_lm.calculate_prob_of_sentence(filled_sent)
# #
# #         results.append((orig,masked,filled_sent,pred_tokens,pp_plen,pp_comm))
# #
# #     with open(os.path.join(output_dir,"txt.results_sents_sampled"),"w",encoding="utf-8") as f:
# #         for r in results:
# #             f.write(f"original_sentence: {r[0]}\n")
# #             f.write(f"masked_sentence: {r[1]}\n")
# #             f.write(f"plenary_sentence: {r[2]}\n")
# #             f.write("plenary_tokens: "+",".join(r[3])+"\n")
# #             f.write(f"probability of plenary sentence in plenary corpus: {r[4]:.2f}\n")
# #             f.write(f"probability of plenary sentence in committee corpus: {r[5]:.2f}\n")
# #
# #     # perplexity on masked tokens
# #     log_probs=[]
# #     total_masked=0
# #     for (orig,masked,filled,pred_tokens,pp_plen,pp_comm) in results:
# #         om_toks=masked.split()
# #         fi_toks=filled.split()
# #         f_tokens=["_s0_","_s1_"]+fi_toks
# #         for i,tk in enumerate(om_toks):
# #             if tk=="[*]":
# #                 pos=i+2
# #                 w1,w2,w3=f_tokens[pos-2], f_tokens[pos-1], f_tokens[pos]
# #                 p=plenary_lm._interp_prob(w1,w2,w3)
# #                 if p<=0:
# #                     p=1e-15
# #                 log_probs.append(math.log2(p))
# #                 total_masked+=1
# #     if total_masked==0:
# #         pp=0.0
# #     else:
# #         avg_logp=sum(log_probs)/total_masked
# #         pp=2**(-avg_logp)
# #
# #     with open(os.path.join(output_dir,"txt.result_perplexity"),"w",encoding="utf-8") as f:
# #         f.write(f"{pp:.2f}\n")
# #
# # if __name__=="__main__":
# #     main()
#
#



# 12/12/24, 19:37:

# import json
# import math
# import random
# import os
# import sys
# import collections
# import pandas as pd
#
# # this code implements the instructions of the assignment as previously discussed.
# # changes now:
# # - we have a global comma_penalty factor to penalize selecting a comma
# # - we must not predict "_s0_" or "_s1_" tokens at all. these are placeholders only.
# # - the user said perplexity is still bad; we try to ensure we pick proper tokens by excluding these placeholders from predictions.
# #
# # steps:
# # 1. read corpus with "protocol_type" and "sentence_text"
# # 2. build trigram models for committee and plenary
# # 3. extract collocations
# # 4. pick 10 committee sentences with >=5 tokens, mask 10% of tokens
# # 5. predict masked tokens using plenary model, apply comma penalty, exclude "_s0_","_s1_" from predictions
# # 6. compute perplexity on masked tokens
# #
# # we just implement what the assignment requires, no iterative search now.
#
# comma_penalty = 0.973  # global variable, can be updated manually
#
# def tokenize_sentence(sent):
#     # splits sentence by whitespace
#     return sent.strip().split()
#
# class LM_Trigram:
#     def __init__(self, sentences, lambdas=(0.695,0.154,0.151)):
#         # builds trigram model with given lambdas for interpolation
#         self.l1, self.l2, self.l3 = lambdas
#         self.unigram_counts = collections.Counter()
#         self.bigram_counts = collections.Counter()
#         self.trigram_counts = collections.Counter()
#
#         for sent in sentences:
#             s = ["_s0_", "_s1_"] + sent
#             for w in s:
#                 self.unigram_counts[w]+=1
#             for i in range(len(s)-1):
#                 self.bigram_counts[(s[i],s[i+1])]+=1
#             for i in range(len(s)-2):
#                 self.trigram_counts[(s[i],s[i+1],s[i+2])]+=1
#
#         self.v = len(self.unigram_counts)
#         self.total_unigrams = sum(self.unigram_counts.values())
#
#     def _laplace_unigram(self, w):
#         return (self.unigram_counts[w]+1)/(self.total_unigrams+self.v)
#
#     def _laplace_bigram(self, w1, w2):
#         count_bi=self.bigram_counts[(w1,w2)]
#         count_uni=self.unigram_counts[w1]
#         return (count_bi+1)/(count_uni+self.v)
#
#     def _laplace_trigram(self, w1, w2, w3):
#         count_tri=self.trigram_counts[(w1,w2,w3)]
#         count_bi=self.bigram_counts[(w1,w2)]
#         return (count_tri+1)/(count_bi+self.v)
#
#     def _interp_prob(self, w1,w2,w3):
#         p_uni=self._laplace_unigram(w3)
#         p_bi=self._laplace_bigram(w2,w3)
#         p_tri=self._laplace_trigram(w1,w2,w3)
#         return self.l1*p_uni + self.l2*p_bi + self.l3*p_tri
#
#     def calculate_prob_of_sentence(self, sentence):
#         # calculates log probability of sentence
#         toks=sentence.strip().split()
#         toks=["_s0_","_s1_"]+toks
#         log_prob=0.0
#         for i in range(2,len(toks)):
#             p=self._interp_prob(toks[i-2],toks[i-1],toks[i])
#             log_prob+=math.log(p)
#         return log_prob
#
#     def generate_next_token(self, context):
#         # predicts next token given context
#         # exclude "_s0_" and "_s1_" from predictions
#         # apply comma penalty if w3==','
#         ctx=context.strip().split()
#         if len(ctx)<2:
#             ctx=["_s0_","_s1_"]+ctx
#         w1,w2=ctx[-2],ctx[-1]
#         max_token=None
#         max_prob=-1
#         for w3 in self.unigram_counts:
#             if w3 in ("_s0_","_s1_"):
#                 continue
#             p=self._interp_prob(w1,w2,w3)
#             if w3==',':
#                 p*=comma_penalty
#             if p>max_prob:
#                 max_prob=p
#                 max_token=w3
#         return (max_token, math.log(max_prob))
#
# def get_k_n_t_collocations(k, n, t, corpus_df, metric_type="frequency"):
#     # returns top k collocations of length n with at least t appearances
#     docs = corpus_df["sentence_text"].tolist()
#     ngram_doc_freq={}
#     for i,doc in enumerate(docs):
#         tokens=tokenize_sentence(doc)
#         ngs=zip(*[tokens[j:] for j in range(n)])
#         doc_count=collections.Counter(ngs)
#         for ng,c in doc_count.items():
#             if ng not in ngram_doc_freq:
#                 ngram_doc_freq[ng]={}
#             ngram_doc_freq[ng][i]=c
#
#     global_count={ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
#     filtered={ng:freq for ng,freq in global_count.items() if freq>=t}
#
#     if metric_type=="frequency":
#         sorted_ngrams=sorted(filtered.items(), key=lambda x:x[1], reverse=True)
#         return [(" ".join(ng),val) for ng,val in sorted_ngrams[:k]]
#     elif metric_type=="tfidf":
#         D=len(docs)
#         doc_appearances={ng: len(ngram_doc_freq[ng]) for ng in filtered}
#         scores={}
#         for ng in filtered:
#             df_count=doc_appearances[ng]
#             if df_count==0:
#                 continue
#             idf=math.log(D/df_count)
#             total_score=0.0
#             for d_i,freq_ng in ngram_doc_freq[ng].items():
#                 tokens_count=len(tokenize_sentence(docs[d_i]))
#                 tf=freq_ng/tokens_count if tokens_count>0 else 0
#                 total_score+=tf*idf
#             scores[ng]=total_score
#         sorted_scores=sorted(scores.items(), key=lambda x:x[1], reverse=True)
#         return [(" ".join(ng),sc) for ng,sc in sorted_scores[:k]]
#     else:
#         return []
#
# def mask_tokens_in_sentences(sentences,x):
#     # masks x% tokens in each sentence
#     masked=[]
#     for s in sentences:
#         toks=s.strip().split()
#         if len(toks)==0:
#             masked.append(s)
#             continue
#         num_to_mask=max(1,int(len(toks)*x/100))
#         if num_to_mask>len(toks):
#             num_to_mask=len(toks)
#         idxs=random.sample(range(len(toks)),num_to_mask)
#         for i in idxs:
#             toks[i]="[*]"
#         masked.append(" ".join(toks))
#     return masked
#
# def main():
#     corpus_file=sys.argv[1]
#     output_dir=sys.argv[2]
#
#     data=[]
#     # reading corpus with fields protocol_type and sentence_text
#     with open(corpus_file,'r',encoding='utf-8') as f:
#         for line in f:
#             line=line.strip()
#             if not line:
#                 continue
#             obj=json.loads(line)
#             t=obj['protocol_type'].strip().lower()
#             txt=obj['sentence_text'].strip()
#             data.append((t,txt))
#     df=pd.DataFrame(data,columns=["protocol_type","sentence_text"])
#
#     committee_df=df[df.protocol_type=="committee"].copy()
#     plenary_df=df[df.protocol_type=="plenary"].copy()
#
#     committee_sents=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
#     plenary_sents=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]
#
#     # choose lambdas as per instructions
#     l1,l2,l3=0.1,0.3,0.6
#     committee_lm=LM_Trigram(committee_sents,(l1,l2,l3))
#     plenary_lm=LM_Trigram(plenary_sents,(l1,l2,l3))
#
#     # collocations
#     with open(os.path.join(output_dir,"txt.collocations_knesset"),"w",encoding="utf-8") as f:
#         for n in [2,3,4]:
#             if n==2:
#                 f.write("Two-gram collocations:\n")
#             elif n==3:
#                 f.write("Three-gram collocations:\n")
#             else:
#                 f.write("Four-gram collocations:\n")
#
#             f.write("Frequency:\nCommittee corpus:\n")
#             freq_comm = get_k_n_t_collocations(10,n,5,committee_df,"frequency")
#             for c_val in freq_comm:
#                 f.write(c_val[0]+"\n")
#             f.write("\nPlenary corpus:\n")
#             freq_plen = get_k_n_t_collocations(10,n,5,plenary_df,"frequency")
#             for p_val in freq_plen:
#                 f.write(p_val[0]+"\n")
#             f.write("\n")
#
#             f.write("TF-IDF:\nCommittee corpus:\n")
#             tfidf_comm = get_k_n_t_collocations(10,n,5,committee_df,"tfidf")
#             for c_val in tfidf_comm:
#                 f.write(c_val[0]+"\n")
#             f.write("\nPlenary corpus:\n")
#             tfidf_plen = get_k_n_t_collocations(10,n,5,plenary_df,"tfidf")
#             for p_val in tfidf_plen:
#                 f.write(p_val[0]+"\n")
#             f.write("\n")
#
#     committee_texts=committee_df["sentence_text"].tolist()
#     candidate_sents=[s for s in committee_texts if len(tokenize_sentence(s))>=5]
#     random.seed(42)
#     sampled_sents=random.sample(candidate_sents,10)
#     masked_sents=mask_tokens_in_sentences(sampled_sents,10)
#
#     with open(os.path.join(output_dir,"txt.sents_sampled_original"),"w",encoding="utf-8") as f:
#         for s in sampled_sents:
#             f.write(s+"\n")
#
#     with open(os.path.join(output_dir,"txt.sents_sampled_masked"),"w",encoding="utf-8") as f:
#         for s in masked_sents:
#             f.write(s+"\n")
#
#     results=[]
#     # fill masked tokens using plenary model, with comma penalty and exclude _s0_/_s1_
#     for orig,masked in zip(sampled_sents,masked_sents):
#         om_toks=masked.split()
#         fi_toks=om_toks[:]
#         pred_tokens=[]
#         for i,tk in enumerate(fi_toks):
#             if tk=="[*]":
#                 ctx=" ".join(fi_toks[:i])
#                 # predict next token
#                 # model's generate_next_token returns top token and log prob, but we must do something special
#                 # no, we already handle punctuation inside generate_next_token as we added comma penalty there
#                 # just call generate_next_token:
#                 next_tk,next_log_p=plenary_lm.generate_next_token(ctx)
#                 fi_toks[i]=next_tk
#                 pred_tokens.append(next_tk)
#         filled_sent=" ".join(fi_toks)
#
#         pp_plen=plenary_lm.calculate_prob_of_sentence(filled_sent)
#         pp_comm=committee_lm.calculate_prob_of_sentence(filled_sent)
#         results.append((orig,masked,filled_sent,pred_tokens,pp_plen,pp_comm))
#
#     with open(os.path.join(output_dir,"txt.results_sents_sampled"),"w",encoding="utf-8") as f:
#         for r in results:
#             f.write(f"original_sentence: {r[0]}\n")
#             f.write(f"masked_sentence: {r[1]}\n")
#             f.write(f"plenary_sentence: {r[2]}\n")
#             f.write("plenary_tokens: "+",".join(r[3])+"\n")
#             f.write(f"probability of plenary sentence in plenary corpus: {r[4]:.2f}\n")
#             f.write(f"probability of plenary sentence in committee corpus: {r[5]:.2f}\n")
#
#     # perplexity on masked tokens
#     log_probs=[]
#     total_masked=0
#     for (orig,masked,filled,pred_tokens,pp_plen,pp_comm) in results:
#         om_toks=masked.split()
#         fi_toks=filled.split()
#         f_tokens=["_s0_","_s1_"]+fi_toks
#         for i,tk in enumerate(om_toks):
#             if tk=="[*]":
#                 pos=i+2
#                 w1,w2,w3=f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos]
#                 p=plenary_lm._interp_prob(w1,w2,w3)
#                 # apply comma penalty if comma
#                 if w3==',':
#                     p*=comma_penalty
#                 if p<=0:
#                     p=1e-15
#                 log_probs.append(math.log2(p))
#                 total_masked+=1
#     if total_masked==0:
#         pp=0.0
#     else:
#         avg_logp=sum(log_probs)/total_masked
#         pp=2**(-avg_logp)
#
#     with open(os.path.join(output_dir,"txt.result_perplexity"),"w",encoding="utf-8") as f:
#         f.write(f"{pp:.2f}\n")
#
# if __name__=="__main__":
#     main()
