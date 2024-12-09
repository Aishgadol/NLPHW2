import json
import math
import random
import os
import sys
import collections
import pandas as pd
import re
from math import log

# User wants even better accuracy by giving it more time to compute.
# We'll:
# 1. Keep using Kneser-Ney smoothing for improved perplexity.
# 2. Add caching of probabilities to speed repeated computations, allowing us to not reduce the search space too much.
# 3. Increase the search limit for generation since we can afford more time.
# 4. Include more print statements to track progress.
#
# We'll implement caching for kn_prob_bigram and kn_prob_trigram to avoid recomputing the sums every time.
# With caching, we can safely increase the limit of valid words considered for generation to try more candidates,
# potentially improving accuracy (by not restricting to only top 2000 words, maybe top 5000 or entire vocab).
#
# Note: This code assumes we have enough time and resources. It may still run slowly depending on corpus size.

HEBREW_RE = re.compile(r'[\u0590-\u05FF]+')

def tokenize_sentence(sent):
    return re.findall(HEBREW_RE, sent)

def is_valid_word(w):
    return w and not w.startswith('_') and HEBREW_RE.fullmatch(w) is not None

class KneserNeyTrigramLM:
    def __init__(self, sentences, discount=0.75):
        print("Building Kneser-Ney trigram LM...")
        self.discount = discount
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()
        self.context_of_bigram = collections.defaultdict(set)
        self.context_of_unigram = collections.defaultdict(set)

        for s in sentences:
            s = ["_s0_", "_s1_"] + s
            for w in s:
                self.unigram_counts[w] += 1
            for i in range(len(s)-1):
                self.bigram_counts[(s[i], s[i+1])] += 1
                self.context_of_unigram[s[i+1]].add(s[i])
            for i in range(len(s)-2):
                self.trigram_counts[(s[i], s[i+1], s[i+2])] += 1
                self.context_of_bigram[(s[i+1], s[i+2])].add(s[i])

        self.V = len(self.unigram_counts)
        self.total = sum(self.unigram_counts.values())

        self.bigram_continuation = collections.Counter()
        self.unigram_continuation = collections.Counter()

        for (w1,w2,w3), c in self.trigram_counts.items():
            self.bigram_continuation[(w2,w3)] +=1
        for (w1,w2), c in self.bigram_counts.items():
            self.unigram_continuation[w2]+=1

        self._cached_uni_p = None

        self.valid_words = [w for w in self.unigram_counts if is_valid_word(w)]
        self.valid_words.sort(key=lambda x:self.unigram_counts[x], reverse=True)

        # Precompute bigram_totals and distinct counts
        self.bigram_totals = collections.Counter()
        for (a,b),cnt in self.bigram_counts.items():
            self.bigram_totals[a]+=cnt

        self.trigram_distinct_next = collections.Counter()
        for (w1,w2,w3) in self.trigram_counts:
            self.trigram_distinct_next[(w1,w2)] += 1

        self.bigram_distinct_next = collections.Counter()
        for (w2,w3) in self.bigram_counts:
            self.bigram_distinct_next[w2]+=1

        # Caching dictionaries
        self._cache_bi = {}
        self._cache_tri = {}

        print("KN LM built. Vocab size:", self.V, "Total tokens:", self.total, "Valid words:", len(self.valid_words))

    def kn_prob_unigram(self, w):
        cont = self.unigram_continuation[w]
        if self._cached_uni_p is None:
            total_cont = sum(self.unigram_continuation.values())
            self._cached_uni_p = total_cont
        total_cont = self._cached_uni_p
        if total_cont == 0:
            return 1/self.V
        return cont/total_cont

    def kn_prob_bigram(self, w2, w3):
        key=(w2,w3)
        if key in self._cache_bi:
            return self._cache_bi[key]
        c12 = self.bigram_counts[key]
        c1 = self.bigram_totals[w2]
        if c1==0:
            val=self.kn_prob_unigram(w3)
            self._cache_bi[key]=val
            return val
        d=self.discount
        distinct_next=self.bigram_distinct_next[w2]
        lam=(d*distinct_next)/c1 if c1>0 else 0
        val= max(c12-d,0)/c1 + lam*self.kn_prob_unigram(w3)
        self._cache_bi[key]=val
        return val

    def kn_prob_trigram(self, w1, w2, w3):
        key=(w1,w2,w3)
        if key in self._cache_tri:
            return self._cache_tri[key]

        c123 = self.trigram_counts[key]
        c12 = self.bigram_counts[(w1,w2)]
        if c12==0:
            val=self.kn_prob_bigram(w2,w3)
            self._cache_tri[key]=val
            return val
        d=self.discount
        distinct_next = self.trigram_distinct_next[(w1,w2)]
        lam=(d*distinct_next)/c12
        val= max(c123 - d,0)/c12 + lam*self.kn_prob_bigram(w2,w3)
        self._cache_tri[key]=val
        return val

    def sentence_logprob(self, sent):
        toks=sent.strip().split()
        toks=["_s0_","_s1_"]+toks
        lp=0.0
        for i in range(2,len(toks)):
            p=self.kn_prob_trigram(toks[i-2],toks[i-1],toks[i])
            if p<=0:
                p=1e-15
            lp+=math.log(p)
        return lp

    def generate_next_token(self, context):
        ctx=context.strip().split()
        if len(ctx)<2:
            ctx=["_s0_","_s1_"]+ctx
        w1,w2=ctx[-2],ctx[-1]

        # We'll try a larger limit now that we have caching
        limit=min(len(self.valid_words),10000)
        best_tk=None
        best_p=-1
        print("         Checking up to", limit, "candidate words...")
        for idx,w3 in enumerate(self.valid_words[:limit]):
            p=self.kn_prob_trigram(w1,w2,w3)
            if p>best_p:
                best_p=p
                best_tk=w3
            if (idx+1)%2000==0:
                print("         ...checked", idx+1, "words")
        return best_tk,best_p

def load_corpus(jsonl_path):
    data=[]
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            tp=obj['type_protocol'].strip().lower()
            txt=obj['text_sentence']
            data.append((tp,txt))
    df=pd.DataFrame(data,columns=["type_protocol","text_sentence"])
    return df

def mask_tokens_in_sentences(sents, x):
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
    pred=[]
    valid=[w for w in lm.unigram_counts if is_valid_word(w)]
    fallback="שלום"
    if valid:
        fallback=max(valid, key=lambda w: lm.unigram_counts[w])

    print("   Filling a masked sentence with plenary model... tokens:", len(toks))
    masked_positions=[i for i,tk in enumerate(toks) if tk=="[*]"]
    print("   Masked positions:", masked_positions)
    for count, i in enumerate(masked_positions, start=1):
        print("      Predicting token at masked index", i, f"({count}/{len(masked_positions)})")
        ctx=" ".join(toks[:i])
        ptk,p=lm.generate_next_token(ctx)
        if ptk is None:
            ptk=fallback
        toks[i]=ptk
        pred.append(ptk)
    print("   Done filling this sentence.")
    return " ".join(toks), pred

def compute_perplexity_of_masked_tokens(lm,orig_masked,filled):
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
                    p=lm.kn_prob_trigram(f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos])
                    if p<=0:
                        p=1e-15
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
        print("No input file.")
        sys.exit(1)

    print("Loading corpus...")
    df=load_corpus(json_file)
    committee_df=df[df.type_protocol=="committee"].copy()
    plenary_df=df[df.type_protocol=="plenary"].copy()

    print("Building committee KN LM...")
    committee_sents=[tokenize_sentence(s) for s in committee_df["text_sentence"]]
    committee_lm=KneserNeyTrigramLM(committee_sents, discount=0.75)

    print("Building plenary KN LM...")
    plenary_sents=[tokenize_sentence(s) for s in plenary_df["text_sentence"]]
    plenary_lm=KneserNeyTrigramLM(plenary_sents, discount=0.75)

    print("Sampling committee sentences...")
    c_texts=committee_df["text_sentence"].tolist()
    sample_size=min(10,len(c_texts))
    if sample_size==0:
        print("No committee sents.")
        sys.exit(1)
    random.seed(42)
    sampled_raw=random.sample(c_texts,sample_size)
    sampled=[" ".join(tokenize_sentence(s)) for s in sampled_raw if tokenize_sentence(s)]
    if not sampled:
        print("No valid tokenized sentences after filtering.")
        sys.exit(1)

    print("Masking tokens...")
    masked=mask_tokens_in_sentences(sampled,10)

    print("Writing original and masked sentences...")
    with open("sents_sampled_original.txt","w",encoding="utf-8") as f:
        for s in sampled:
            f.write(s+"\n")
    with open("sents_sampled_masked.txt","w",encoding="utf-8") as f:
        for s in masked:
            f.write(s+"\n")

    print("Filling masked tokens using plenary model...")
    results=[]
    for orig,m in zip(sampled,masked):
        print("   Processing sentence:", (orig[:50]+"...") if len(orig)>50 else orig)
        filled,pred=fill_masked_tokens(m,plenary_lm)
        pp_plen=plenary_lm.sentence_logprob(filled)
        pp_comm=committee_lm.sentence_logprob(filled)
        results.append((orig,m,filled,pred,pp_plen,pp_comm))

    print("Writing results...")
    with open("results_sents_sampled.txt","w",encoding="utf-8") as f:
        for r in results:
            f.write(f"original_sentence: {r[0]}\n")
            f.write(f"masked_sentence: {r[1]}\n")
            f.write(f"plenary_sentence: {r[2]}\n")
            f.write("plenary_tokens: "+",".join(r[3])+"\n")
            f.write(f"probability of plenary sentence in plenary corpus: {r[4]:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {r[5]:.2f}\n")

    print("Computing perplexity...")
    pp=compute_perplexity_of_masked_tokens(plenary_lm,masked,[x[2] for x in results])
    with open("result_perplexity.txt","w",encoding="utf-8") as f:
        f.write(f"{pp:.2f}\n")

    print("Done.")

if __name__=="__main__":
    main()
