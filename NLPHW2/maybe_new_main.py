import json
import math
import random
import os
import sys
import collections
import pandas as pd
import re

# Revised solution integrating previous lessons and hints from the provided code.
# Key points:
# - Still rely on Laplace smoothing and linear interpolation only.
# - Strong trigram weight to improve context accuracy.
# - Tokenization strictly on Hebrew words.
# - Avoid predicting rare or non-Hebrew words by:
#   * Choosing next tokens only from a filtered set of frequently occurring Hebrew words.
#   * This should improve accuracy and perplexity since we avoid odd or rare guesses.
# - Print progress steps.
# - Keep code structure clean and natural.

HEBREW_RE = re.compile(r'[\u0590-\u05FF]+')

def tokenize_sentence(sent):
    return re.findall(HEBREW_RE, sent)

def is_valid_word(w):
    return w and not w.startswith('_') and HEBREW_RE.fullmatch(w) is not None

class LM_Trigram:
    def __init__(self, sentences, lambdas=(0.1,0.2,0.7)):
        print("Building LM...")
        self.l1, self.l2, self.l3 = lambdas
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()
        self._build_counts(sentences)
        # Precompute a list of valid words sorted by frequency
        self.valid_words = [w for w in self.unigram_counts if is_valid_word(w)]
        self.valid_words.sort(key=lambda x: self.unigram_counts[x], reverse=True)
        print("LM built. Valid words:", len(self.valid_words), "Total vocab:", len(self.unigram_counts))

    def _build_counts(self, sentences):
        for s in sentences:
            s = ["_s0_", "_s1_"] + s
            for w in s:
                self.unigram_counts[w] += 1
            for i in range(len(s)-1):
                self.bigram_counts[(s[i], s[i+1])] += 1
            for i in range(len(s)-2):
                self.trigram_counts[(s[i], s[i+1], s[i+2])] += 1
        self.V = len(self.unigram_counts)
        self.total = sum(self.unigram_counts.values())

    def _laplace_uni(self, w):
        return (self.unigram_counts[w] + 1)/(self.total + self.V)

    def _laplace_bi(self, w1, w2):
        return (self.bigram_counts[(w1,w2)] + 1)/(self.unigram_counts[w1] + self.V)

    def _laplace_tri(self, w1, w2, w3):
        return (self.trigram_counts[(w1,w2,w3)] + 1)/(self.bigram_counts[(w1,w2)] + self.V)

    def _interp_prob(self, w1, w2, w3):
        p_uni = self._laplace_uni(w3)
        p_bi = self._laplace_bi(w2, w3)
        p_tri = self._laplace_tri(w1, w2, w3)
        return self.l1*p_uni + self.l2*p_bi + self.l3*p_tri

    def calculate_prob_of_sentence(self, sentence):
        toks = sentence.strip().split()
        toks = ["_s0_","_s1_"] + toks
        logp=0.0
        for i in range(2,len(toks)):
            logp += math.log(self._interp_prob(toks[i-2], toks[i-1], toks[i]))
        return logp

    def generate_next_token(self, context):
        # Predict from the top portion of valid words to avoid ultra-rare words
        ctx = context.strip().split()
        if len(ctx)<2:
            ctx=["_s0_","_s1_"]+ctx
        w1,w2=ctx[-2],ctx[-1]
        best=None
        best_p=-1
        # Limit to top 10k frequent words if large, for efficiency and accuracy
        limit = min(len(self.valid_words), 10000)
        for w3 in self.valid_words[:limit]:
            p=self._interp_prob(w1,w2,w3)
            if p>best_p:
                best_p=p
                best=w3
        return (best,best_p)

def read_corpus(path):
    print("Reading corpus:", path)
    data=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            tp=obj["type_protocol"].strip()
            if tp=="ועדה":
                tp="committee"
            elif tp=="מליאה":
                tp="plenary"
            txt=obj["text_sentence"]
            data.append((tp,txt))
    df=pd.DataFrame(data,columns=["type_protocol","text_sentence"])
    print("Committee:", len(df[df.type_protocol=="committee"]), "Plenary:", len(df[df.type_protocol=="plenary"]))
    return df[df.type_protocol=="committee"].copy(), df[df.type_protocol=="plenary"].copy()

def get_k_n_t_collocations(k,n,t,corpus_df,metric_type="frequency"):
    docs=corpus_df["text_sentence"].tolist()
    ngram_doc_freq={}
    for i,doc in enumerate(docs):
        tokens=tokenize_sentence(doc)
        ngs=list(zip(*[tokens[j:] for j in range(n)]))
        c=collections.Counter(ngs)
        for ng,count in c.items():
            if ng not in ngram_doc_freq:
                ngram_doc_freq[ng]={}
            ngram_doc_freq[ng][i]=count
    global_count={ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
    filtered={ng:freq for ng,freq in global_count.items() if freq>=t}
    if metric_type=="frequency":
        sorted_ngrams=sorted(filtered.items(), key=lambda x:x[1], reverse=True)
        return [(" ".join(x[0]),x[1]) for x in sorted_ngrams[:k]]
    elif metric_type=="tfidf":
        D=len(docs)
        if D==0:
            return []
        doc_count_for_ng={ng: len(ngram_doc_freq[ng]) for ng in filtered}
        scores={}
        for ng in filtered:
            df_count=doc_count_for_ng[ng]
            if df_count==0:
                continue
            idf=math.log(D/df_count)
            total=0.0
            for d_i,v in ngram_doc_freq[ng].items():
                toks=tokenize_sentence(docs[d_i])
                all_ngr=list(zip(*[toks[j:] for j in range(n)]))
                total_terms=len(all_ngr)
                tf=v/total_terms if total_terms>0 else 0
                total+=tf*idf
            scores[ng]=total
        sorted_sc=sorted(scores.items(), key=lambda x:x[1], reverse=True)
        return [(" ".join(x[0]), x[1]) for x in sorted_sc[:k]]
    else:
        return []

def mask_tokens_in_sentences(sents, x):
    print(f"Masking {x}% of tokens...")
    masked=[]
    for s in sents:
        toks=s.strip().split()
        if not toks:
            masked.append("")
            continue
        num_to_mask=max(1,int(len(toks)*x/100))
        if num_to_mask>len(toks):
            num_to_mask=len(toks)
        idxs=random.sample(range(len(toks)),num_to_mask)
        for i in idxs:
            toks[i]="[*]"
        masked.append(" ".join(toks))
    return masked

def fill_masked_tokens(sent,lm):
    toks=sent.strip().split()
    predicted=[]
    valid=[w for w in lm.unigram_counts if is_valid_word(w)]
    fallback="שלום"
    if valid:
        fallback=max(valid, key=lambda w: lm.unigram_counts[w])
    for i,tk in enumerate(toks):
        if tk=="[*]":
            ctx=" ".join(toks[:i])
            ptk,p=lm.generate_next_token(ctx)
            if ptk is None:
                ptk=fallback
            toks[i]=ptk
            predicted.append(ptk)
    return " ".join(toks),predicted

def compute_perplexity_of_masked_tokens(lm, orig_masked, filled):
    print("Computing perplexity on masked tokens...")
    log_probs=[]
    total_masked=0
    for om,fi in zip(orig_masked,filled):
        omt=om.strip().split()
        fit=fi.strip().split()
        f_tokens=["_s0_","_s1_"]+fit
        for i,tk in enumerate(omt):
            if tk=="[*]":
                pos=i+2
                if pos<len(f_tokens):
                    w1,w2,w3=f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos]
                    p=lm._interp_prob(w1,w2,w3)
                    log_probs.append(math.log(p,2))
                    total_masked+=1
    if total_masked==0:
        return 0.0
    avg=sum(log_probs)/total_masked
    pp=2**(-avg)
    return pp

def main():
    json_file="result.jsonl"
    if not os.path.exists(json_file):
        print("result.jsonl not found.")
        sys.exit(1)

    print("Loading corpus...")
    committee_df, plenary_df=read_corpus(json_file)

    print("Building committee model...")
    committee_sents=[tokenize_sentence(s) for s in committee_df["text_sentence"]]
    committee_lm=LM_Trigram(committee_sents,(0.1,0.2,0.7))

    print("Building plenary model...")
    plenary_sents=[tokenize_sentence(s) for s in plenary_df["text_sentence"]]
    plenary_lm=LM_Trigram(plenary_sents,(0.1,0.2,0.7))

    print("Extracting collocations...")
    with open("collocations_knesset.txt","w",encoding="utf-8") as f:
        for n in [2,3,4]:
            if n==2:
                f.write("Two-gram collocations:\n")
            elif n==3:
                f.write("Three-gram collocations:\n")
            else:
                f.write("Four-gram collocations:\n")

            f.write("Frequency:\nCommittee corpus:\n")
            freq_c=get_k_n_t_collocations(10,n,5,committee_df,"frequency")
            for cval in freq_c:
                f.write(cval[0]+"\n")
            f.write("\nPlenary corpus:\n")
            freq_p=get_k_n_t_collocations(10,n,5,plenary_df,"frequency")
            for pval in freq_p:
                f.write(pval[0]+"\n")
            f.write("\nTF-IDF:\nCommittee corpus:\n")
            tfidf_c=get_k_n_t_collocations(10,n,5,committee_df,"tfidf")
            for cval in tfidf_c:
                f.write(cval[0]+"\n")
            f.write("\nPlenary corpus:\n")
            tfidf_p=get_k_n_t_collocations(10,n,5,plenary_df,"tfidf")
            for pval in tfidf_p:
                f.write(pval[0]+"\n")
            f.write("\n")

    print("Sampling committee sentences...")
    c_texts=committee_df["text_sentence"].tolist()
    sample_size=min(10,len(c_texts))
    if sample_size==0:
        print("No committee sentences.")
        sys.exit(1)
    random.seed(142)
    sampled_raw=random.sample(c_texts,sample_size)
    sampled=[" ".join(tokenize_sentence(s)) for s in sampled_raw]
    masked=mask_tokens_in_sentences(sampled,10)

    with open("sents_sampled_original.txt","w",encoding="utf-8") as f:
        for s in sampled:
            f.write(s+"\n")

    with open("sents_sampled_masked.txt","w",encoding="utf-8") as f:
        for s in masked:
            f.write(s+"\n")

    print("Filling masked tokens using plenary model...")
    results=[]
    for orig,msk in zip(sampled,masked):
        filled,pred=fill_masked_tokens(msk,plenary_lm)
        pp_plen=plenary_lm.calculate_prob_of_sentence(filled)
        pp_comm=committee_lm.calculate_prob_of_sentence(filled)
        results.append((orig,msk,filled,pred,pp_plen,pp_comm))

    with open("results_sents_sampled.txt","w",encoding="utf-8") as f:
        for r in results:
            f.write(f"original_sentence: {r[0]}\n")
            f.write(f"masked_sentence: {r[1]}\n")
            f.write(f"plenary_sentence: {r[2]}\n")
            f.write("plenary_tokens: "+",".join(r[3])+"\n")
            f.write(f"probability of plenary sentence in plenary corpus: {r[4]:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {r[5]:.2f}\n")

    print("Calculating perplexity...")
    pp=compute_perplexity_of_masked_tokens(plenary_lm,masked,[x[2] for x in results])
    with open("result_perplexity.txt","w",encoding="utf-8") as f:
        f.write(f"{pp:.2f}\n")

    print("Done. Check outputs.")

if __name__=="__main__":
    main()
