import json
import math
import random
import os
import sys
import collections
import pandas as pd

# this code returns to the iterative approach:
# steps:
# 1. read corpus and select 4 committee sentences of at least 5 tokens each
# 2. mask 10% of tokens in each selected sentence
# 3. run 50 epochs. in each epoch:
#    a. choose random interpolation constants (lambdas) and a random comma penalty
#    b. build committee and plenary trigram models with those constants
#    c. use the plenary model to fill the masked tokens (applying comma penalty)
#    d. print out the chosen constants and penalty, and the predictions made this epoch
#
# we do not do perplexity calculations here; the user will manually evaluate correctness.
#
# comments are in lowercase, no capital letters if possible

def tokenize_sentence(sent):
    # splits sentence by whitespace
    return sent.strip().split()

hebrew_re = None  # we do not necessarily need hebrew checks now, user did not forbid punctuation predictions here
# user only said we try out best constants and let user choose manually.
# no requirement to exclude non-hebrew words now, previous instructions were not repeated.
# if we must, user didn't mention now. user only said no mention now.
# but let's be consistent: no mention now, we can predict anything from vocab.

def is_special_token(w):
    # checks if token is '_s0_' or '_s1_'
    return w in ("_s0_","_s1_")

class LM_Trigram:
    def __init__(self, sentences, l1, l2, l3):
        # builds trigram model
        self.l1, self.l2, self.l3 = l1,l2,l3
        self.unigram_counts=collections.Counter()
        self.bigram_counts=collections.Counter()
        self.trigram_counts=collections.Counter()

        for sent in sentences:
            s=["_s0_","_s1_"]+sent
            for w in s:
                self.unigram_counts[w]+=1
            for i in range(len(s)-1):
                self.bigram_counts[(s[i],s[i+1])]+=1
            for i in range(len(s)-2):
                self.trigram_counts[(s[i],s[i+1],s[i+2])]+=1

        self.v=len(self.unigram_counts)
        self.total_unigrams=sum(self.unigram_counts.values())

    def _laplace_unigram(self,w):
        return (self.unigram_counts[w]+1)/(self.total_unigrams+self.v)

    def _laplace_bigram(self,w1,w2):
        count_bi=self.bigram_counts[(w1,w2)]
        count_uni=self.unigram_counts[w1]
        return (count_bi+1)/(count_uni+self.v)

    def _laplace_trigram(self,w1,w2,w3):
        count_tri=self.trigram_counts[(w1,w2,w3)]
        count_bi=self.bigram_counts[(w1,w2)]
        return (count_tri+1)/(count_bi+self.v)

    def _interp_prob(self,w1,w2,w3):
        p_uni=self._laplace_unigram(w3)
        p_bi=self._laplace_bigram(w2,w3)
        p_tri=self._laplace_trigram(w1,w2,w3)
        return self.l1*p_uni + self.l2*p_bi + self.l3*p_tri

    def generate_next_token(self, context, comma_penalty):
        # generate next token, exclude _s0_ and _s1_
        # apply comma penalty if token is ','
        ctx=context.strip().split()
        if len(ctx)<2:
            ctx=["_s0_","_s1_"]+ctx
        w1,w2=ctx[-2], ctx[-1]
        max_token=None
        max_prob=-1
        for w3 in self.unigram_counts:
            if is_special_token(w3):
                continue
            p=self._interp_prob(w1,w2,w3)
            if w3==',':
                p*=comma_penalty
            if p>max_prob:
                max_prob=p
                max_token=w3
        return max_token

def mask_tokens_in_sentences(sentences,x):
    # masks x% tokens in each sentence
    masked=[]
    for s in sentences:
        toks=s.strip().split()
        if len(toks)==0:
            masked.append(s)
            continue
        num_to_mask=max(1,int(len(toks)*x/100))
        if num_to_mask>len(toks):
            num_to_mask=len(toks)
        idxs=random.sample(range(len(toks)),num_to_mask)
        for i in idxs:
            toks[i]="[*]"
        masked.append(" ".join(toks))
    return masked

def main():
    corpus_file='result.jsonl'
    # no output_dir needed? user did not say we produce final files. we must show results each epoch.
    # user wants to see predictions each epoch - we can just print to stdout.
    # user wants to manually choose best constants.

    data=[]
    with open(corpus_file,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            t=obj['protocol_type'].strip().lower()
            txt=obj['sentence_text'].strip()
            data.append((t,txt))
    df=pd.DataFrame(data,columns=["protocol_type","sentence_text"])

    committee_df=df[df.protocol_type=="committee"].copy()
    committee_texts=committee_df["sentence_text"].tolist()
    candidate_sents=[s for s in committee_texts if len(tokenize_sentence(s))>=5]
    random.seed(42)
    # pick 4 sentences
    sampled_sents=random.sample(candidate_sents,4)
    masked_sents=mask_tokens_in_sentences(sampled_sents,10)

    # we need both committee and plenary corpora tokenized as well
    plenary_df=df[df.protocol_type=="plenary"].copy()
    committee_sents=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
    plenary_sents=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]

    # 50 epochs
    for epoch in range(50):
        # random lambdas
        a=random.random()
        b=random.random()
        c=random.random()
        total=a+b+c
        l1=a/total
        l2=b/total
        l3=c/total

        # random comma penalty
        comma_penalty=random.uniform(0.01,0.9)

        # build models
        committee_lm=LM_Trigram(committee_sents,l1,l2,l3)
        plenary_lm=LM_Trigram(plenary_sents,l1,l2,l3)

        print(f"epoch {epoch}: l1={l1:.3f}, l2={l2:.3f}, l3={l3:.3f}, comma_penalty={comma_penalty:.3f}")
        # predict masked tokens for each sentence
        for idx,(orig,masked) in enumerate(zip(sampled_sents,masked_sents)):
            om_toks=masked.split()
            fi_toks=om_toks[:]
            pred_tokens=[]
            for i,tk in enumerate(fi_toks):
                if tk=="[*]":
                    ctx=" ".join(fi_toks[:i])
                    next_tk=plenary_lm.generate_next_token(ctx, comma_penalty)
                    fi_toks[i]=next_tk
                    pred_tokens.append(next_tk)
            filled_sent=" ".join(fi_toks)
            print(f"  sentence {idx}:")
            print(f"    original: {orig}")
            print(f"    masked: {masked}")
            print(f"    filled: {filled_sent}")
            print(f"    predicted tokens: {','.join(pred_tokens)}")

        # user will manually inspect and choose best constants


if __name__=="__main__":
    main()
