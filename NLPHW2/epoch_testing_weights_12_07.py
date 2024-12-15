import json
import math
import random
import os
import sys
import collections
import pandas as pd

def tokenize_sentence(sent):
    # splits sentence by whitespace
    return sent.strip().split()

class TrigramLM:
    def __init__(self, sentences, lambdas, punctuation_penalty=0.0):
        # builds trigram model with given lambdas for interpolation and punctuation penalty
        self.l1, self.l2, self.l3 = lambdas
        self.punctuation_penalty = punctuation_penalty
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()

        for s in sentences:
            tokens = ['_s0_', '_s1_'] + tokenize_sentence(s)
            for w in tokens:
                self.unigram_counts[w] += 1
            for i in range(len(tokens)-1):
                self.bigram_counts[(tokens[i], tokens[i+1])] += 1
            for i in range(len(tokens)-2):
                self.trigram_counts[(tokens[i], tokens[i+1], tokens[i+2])] += 1

        self.vocab_size = len(self.unigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())

    def _laplace_smooth(self, count, context_count):
        return (count + 1) / (context_count + self.vocab_size)

    def calculate_prob_of_sentence(self, sentence):
        toks = ['_s0_', '_s1_'] + tokenize_sentence(sentence)
        log_prob=0.0
        for i in range(2, len(toks)):
            w1,w2,w3 = toks[i-2], toks[i-1], toks[i]
            p_uni = self._laplace_smooth(self.unigram_counts[w3], self.total_unigrams)
            p_bi = self._laplace_smooth(self.bigram_counts.get((w2,w3),0), self.unigram_counts.get(w2,0))
            p_tri = self._laplace_smooth(self.trigram_counts.get((w1,w2),{}).get(w3,0), self.bigram_counts.get((w1,w2),0))
            prob = self.l1*p_uni + self.l2*p_bi + self.l3*p_tri
            log_prob += math.log(prob)
        return log_prob

    def generate_next_token(self, context):
        ctx = tokenize_sentence(context)
        if len(ctx)<2:
            ctx=["_s0_","_s1_"]+ctx
        w1,w2=ctx[-2], ctx[-1]
        max_token=None
        max_prob=-1
        for w3 in self.unigram_counts:
            if w3 in ("_s0_","_s1_"):
                continue
            p_uni=(self.unigram_counts[w3]+1)/(self.total_unigrams+self.vocab_size)
            p_bi=(self.bigram_counts.get((w2,w3),0)+1)/(self.unigram_counts.get(w2,0)+self.vocab_size)
            p_tri=(self.trigram_counts.get((w1,w2),{}).get(w3,0)+1)/(self.bigram_counts.get((w1,w2),0)+self.vocab_size)
            prob=self.l1*p_uni+self.l2*p_bi+self.l3*p_tri
            # apply punctuation penalty if w3 is punctuation-like (not alpha)
            if not w3.isalpha() and len(w3)>0:
                prob*= (1 - self.punctuation_penalty)
            if prob>max_prob:
                max_prob=prob
                max_token=w3
        return max_token

def mask_tokens(sentences, mask_ratio):
    masked=[]
    for s in sentences:
        toks=tokenize_sentence(s)
        if len(toks)==0:
            masked.append(s)
            continue
        num_to_mask=max(1,int(len(toks)*mask_ratio))
        if num_to_mask>len(toks):
            num_to_mask=len(toks)
        idxs=random.sample(range(len(toks)),num_to_mask)
        for idx in idxs:
            toks[idx]='[*]'
        masked.append(' '.join(toks))
    return masked

def compute_perplexity(model, masked_sents, filled_sents):
    log_prob_sum=0
    token_count=0
    for masked,filled in zip(masked_sents,filled_sents):
        om_toks=tokenize_sentence(masked)
        fi_toks=['_s0_','_s1_']+tokenize_sentence(filled)
        for i,tk in enumerate(om_toks):
            if tk=='[*]':
                w1,w2,w3=fi_toks[i],fi_toks[i+1],fi_toks[i+2]
                p_uni=(model.unigram_counts[w3]+1)/(model.total_unigrams+model.vocab_size)
                p_bi=(model.bigram_counts.get((w2,w3),0)+1)/(model.unigram_counts.get(w2,0)+model.vocab_size)
                p_tri=(model.trigram_counts.get((w1,w2),{}).get(w3,0)+1)/(model.bigram_counts.get((w1,w2),0)+model.vocab_size)
                prob=model.l1*p_uni+model.l2*p_bi+model.l3*p_tri
                # no penalty in perplexity calculation as instructions not to mention it here
                log_prob_sum+=math.log(prob)
                token_count+=1
    if token_count==0:
        return float('inf')
    return math.exp(-log_prob_sum/token_count)

def main(corpus_file, output_dir):
    # read corpus
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
    df=pd.DataFrame(data,columns=['protocol_type','sentence_text'])

    committee_df=df[df.protocol_type=='committee'].copy()
    plenary_df=df[df.protocol_type=='plenary'].copy()

    committee_corpus=committee_df['sentence_text'].tolist()
    plenary_corpus=plenary_df['sentence_text'].tolist()

    # select 4 sentences
    candidate_sents=[s for s in committee_corpus if len(tokenize_sentence(s))>=5]
    random.seed(42)
    sampled_sents=random.sample(candidate_sents,4)
    masked_sents=mask_tokens(sampled_sents,0.1)

    # we run 30 epochs, each epoch tries random constants
    # store top 5 results
    top_results=[]

    print("starting 40 epochs of random searching for weights and punctuation penalty...")
    for epoch in range(40):
        a,b,c=random.random(),random.random(),random.random()
        total=a+b+c
        l1,l2,l3=a/total,b/total,c/total
        punctuation_penalty=random.uniform(0.0,0.9)

        # build models
        print(f"epoch {epoch+1}: building models with l1={l1:.3f}, l2={l2:.3f}, l3={l3:.3f}, punctuation_penalty={punctuation_penalty:.3f}")
        committee_model=TrigramLM(committee_corpus,(l1,l2,l3),punctuation_penalty)
        plenary_model=TrigramLM(plenary_corpus,(l1,l2,l3),punctuation_penalty)

        print(f"epoch {epoch+1}: filling masked sentences...")
        filled_sents=[]
        predictions=[]
        for orig,ms in zip(sampled_sents,masked_sents):
            om_toks=ms.split()
            fi_toks=om_toks[:]
            pred_tokens=[]
            for i,tk in enumerate(fi_toks):
                if tk=='[*]':
                    ctx=' '.join(fi_toks[:i])
                    next_tk=plenary_model.generate_next_token(ctx)
                    fi_toks[i]=next_tk
                    pred_tokens.append(next_tk)
            filled_sent=' '.join(fi_toks)
            filled_sents.append(filled_sent)
            predictions.append((orig, ms, filled_sent, pred_tokens))

        print(f"epoch {epoch+1}: computing perplexity...")
        pp=compute_perplexity(plenary_model,masked_sents,filled_sents)
        print(f"epoch {epoch+1}: perplexity={pp:.4f}")

        # store result
        result_info={
            'perplexity': pp,
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'punctuation_penalty': punctuation_penalty,
            'predictions': predictions
        }
        top_results.append(result_info)
        # keep top 5
        top_results=sorted(top_results,key=lambda x:x['perplexity'])[:5]

        print(f"epoch {epoch+1}: top results so far (best perplexity={top_results[0]['perplexity']:.4f})")

    # at the end, print top 5 results to a file: top_5_results.txt
    # format:
    # <1st/2nd/3rd/4th/5th>
    # <perplexity value>
    # We will also include weights and punctuation penalty
    # We have only 4 sentences total, just show all 4 from each top result

    place_labels=["1st","2nd","3rd","4th","5th"]
    output_path = os.path.join(output_dir, "top_5_results.txt")  # Use output_dir

    with open(output_path,"w",encoding="utf-8") as f:
        for i,res in enumerate(top_results):
            f.write(f"{place_labels[i]} place\n")
            f.write(f"Perplexity: {res['perplexity']:.4f}\n")
            # Include weights and punctuation penalty
            f.write(f"Weights: l1={res['l1']:.3f}, l2={res['l2']:.3f}, l3={res['l3']:.3f}\n")
            f.write(f"Punctuation Penalty: {res['punctuation_penalty']:.3f}\n")
            f.write("\n")
            # show all 4 predicted sets
            # format: original, masked, filled, predicted tokens
            for (orig,ms,filled,preds) in res['predictions']:
                f.write(f"Original: {orig}\n")
                f.write(f"Masked: {ms}\n")
                f.write(f"Filled: {filled}\n")
                f.write(f"Predicted tokens: {', '.join(preds)}\n\n")
            f.write("\n\n\n")

    print(f"finished. top 5 results written to {output_path}")

if __name__=="__main__":
    if len(sys.argv)<3:
        print("usage: python this_script.py <result.jsonl> <output_dir>")
        sys.exit(1)
    corpus_file=sys.argv[1]
    output_dir=sys.argv[2]

    # ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(corpus_file, output_dir)
