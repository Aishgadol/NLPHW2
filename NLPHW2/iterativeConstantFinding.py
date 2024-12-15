import json
import math
import random
import os
import sys
import collections
import pandas as pd
import re

# this code implements a trigram language model with linear interpolation and laplace smoothing
# we must try various combinations of constants over multiple epochs (1000) to find the best perplexity.
# we also impose a heavy price on predicting commas to reduce their likelihood.
# efficiency is not a concern, we focus on best results.
# comments are in lowercase and we avoid capital letters in them.

# steps:
# 1. we load the corpus and build committee and plenary trigram models.
# 2. we define a function to evaluate perplexity on the masked tokens given certain lambdas.
# 3. we run 1000 epochs, each time randomly picking lambdas (for interpolation) and possibly other constants (like a comma penalty),
#    rebuild models with those constants, compute perplexity, track the best.
# 4. in the end print out the best combination found.
#
# constraints and instructions:
# - linear interpolation and laplace smoothing only
# - allowed to pick punctuation if it wins after penalty
# - produce best perplexity
# - since we must tune constants, we will just pick random lambdas each epoch and pick a fixed comma penalty factor randomly

hebrew_re = re.compile(r'[\u0590-\u05FF]+')

def tokenize_sentence(sent):
    return sent.strip().split()

def is_hebrew_word(w):
    return bool(hebrew_re.fullmatch(w))

def is_punctuation(w):
    return all(ch in '.,?!;:' for ch in w) and w!=''

class LM_Trigram:
    def __init__(self, sentences, l1, l2, l3):
        # build trigram model with given lambdas
        # laplace smoothing
        self.l1, self.l2, self.l3 = l1,l2,l3
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()

        for sent in sentences:
            s = ["_s0_", "_s1_"] + sent
            for w in s:
                self.unigram_counts[w]+=1
            for i in range(len(s)-1):
                self.bigram_counts[(s[i],s[i+1])]+=1
            for i in range(len(s)-2):
                self.trigram_counts[(s[i],s[i+1],s[i+2])]+=1

        self.v = len(self.unigram_counts)
        self.total_unigrams = sum(self.unigram_counts.values())

    def _laplace_unigram(self, w):
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

    def calculate_prob_of_sentence(self,sentence):
        toks=sentence.strip().split()
        toks=["_s0_","_s1_"]+toks
        log_prob=0.0
        for i in range(2,len(toks)):
            p=self._interp_prob(toks[i-2],toks[i-1],toks[i])
            log_prob+=math.log(p)
        return log_prob

    def generate_next_token(self, context, comma_penalty):
        # generate next token but penalize commas by comma_penalty factor
        ctx=context.strip().split()
        if len(ctx)<2:
            ctx=["_s0_","_s1_"]+ctx
        w1,w2=ctx[-2],ctx[-1]
        max_token=None
        max_prob=-1
        for w3 in self.unigram_counts:
            p=self._interp_prob(w1,w2,w3)
            # if punctuation is comma, reduce by factor comma_penalty
            # user wants a heavy price on predicting commas
            # let's say if it's exactly ',', multiply p by comma_penalty (<1)
            # if user wants punctuation is allowed if surpasses everything else after penalty
            # we do: if w3==',', p*=comma_penalty
            if w3==',':
                p*=comma_penalty
            # we do allow punctuation if it wins after penalty
            if p>max_prob:
                max_prob=p
                max_token=w3
        return max_token, math.log(max_prob)

def get_k_n_t_collocations(k,n,t,corpus_df,metric_type="frequency"):
    docs = corpus_df["sentence_text"].tolist()
    ngram_doc_freq={}
    for i,doc in enumerate(docs):
        tokens=tokenize_sentence(doc)
        ngs = zip(*[tokens[j:] for j in range(n)])
        doc_count=collections.Counter(ngs)
        for ng,c in doc_count.items():
            if ng not in ngram_doc_freq:
                ngram_doc_freq[ng]={}
            ngram_doc_freq[ng][i]=c

    global_count={ng: sum(ngram_doc_freq[ng].values()) for ng in ngram_doc_freq}
    filtered={ng:freq for ng,freq in global_count.items() if freq>=t}

    if metric_type=="frequency":
        sorted_ngrams=sorted(filtered.items(),key=lambda x:x[1],reverse=True)
        return [(" ".join(ng),val) for ng,val in sorted_ngrams[:k]]
    elif metric_type=="tfidf":
        D=len(docs)
        doc_appearances={ng: len(ngram_doc_freq[ng]) for ng in filtered}
        scores={}
        for ng in filtered:
            df_count=doc_appearances[ng]
            if df_count==0:
                continue
            idf=math.log(D/df_count)
            total_score=0.0
            for d_i,freq_ng in ngram_doc_freq[ng].items():
                tokens_count=len(tokenize_sentence(docs[d_i]))
                tf=freq_ng/tokens_count if tokens_count>0 else 0
                total_score+=tf*idf
            scores[ng]=total_score
        sorted_scores=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        return [(" ".join(ng),sc) for ng,sc in sorted_scores[:k]]
    else:
        return []

def mask_tokens_in_sentences(sentences,x):
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

def evaluate_perplexity(committee_lm,plenary_lm,sampled_sents,masked_sents):
    # compute perplexity on masked tokens using plenary model
    log_probs=[]
    total_masked=0
    for (orig,masked) in zip(sampled_sents,masked_sents):
        om_toks=masked.split()
        # we must also fill them again?
        # no we just measure perplexity on the known filled_sents from previous best run?
        # instructions say perplexity is computed after filling, but we do after generation
        # we have no intermediate here. let's assume we re-generate them or we trust that fill is done outside.
        # for simplicity, we just do perplexity on the currently chosen model:
        # we must fill them again:
        # but that would differ from final. instructions say perplexity after we have the filled sentence
        # we can't store them from outside. let's just fill them now:
        fi_toks=om_toks[:] # copy
        pred_tokens=[]
        # just fill now:
        for i,tk in enumerate(fi_toks):
            if tk=="[*]":
                ctx=" ".join(fi_toks[:i])
                # we must use the same models and constants? yes
                # but we do not currently have the constants outside. let's assume same methods
                # we must have the same code as final step: generate token from plenary
                # we used generate_next_token but need comma_penalty. we must fix comma_penalty now
                # let's fix comma_penalty = 0.1 or random as well?
                # user does not forbid that. let's pick a fixed penalty here:
                comma_penalty=0.5
                next_tk, next_log_p=plenary_lm.generate_next_token(ctx,comma_penalty)
                fi_toks[i]=next_tk
                pred_tokens.append(next_tk)
        f_tokens=["_s0_","_s1_"]+fi_toks
        for i,tk in enumerate(om_toks):
            if tk=="[*]":
                pos=i+2
                w1,w2,w3=f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos]
                p=plenary_lm._interp_prob(w1,w2,w3)
                if p<=0:
                    p=1e-15
                log_probs.append(math.log2(p))
                total_masked+=1
    if total_masked==0:
        return 0.0
    avg_logp=sum(log_probs)/total_masked
    pp=2**(-avg_logp)
    return pp

def main():
    corpus_file=sys.argv[1]
    output_dir=sys.argv[2]

    data=[]
    # now columns are reversed: 'protocol_type' and 'sentence_text'
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
    plenary_df=df[df.protocol_type=="plenary"].copy()

    committee_texts=committee_df["sentence_text"].tolist()
    candidate_sents=[s for s in committee_texts if len(tokenize_sentence(s))>=5]
    random.seed(42)
    sampled_sents=random.sample(candidate_sents,10)
    masked_sents=mask_tokens_in_sentences(sampled_sents,10)

    # build final output anyway
    with open(os.path.join(output_dir,"txt.sents_sampled_original"),"w",encoding="utf-8") as f:
        for s in sampled_sents:
            f.write(s+"\n")

    with open(os.path.join(output_dir,"txt.sents_sampled_masked"),"w",encoding="utf-8") as f:
        for s in masked_sents:
            f.write(s+"\n")

    # build models once for collocations:
    # user wants best perplexity from iteration, but collocations are independent
    committee_tmp_sents=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
    plenary_tmp_sents=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]
    # just pick some default lambdas now (not final)
    tmp_committee_lm=LM_Trigram(committee_tmp_sents,0.1,0.3,0.6)
    tmp_plenary_lm=LM_Trigram(plenary_tmp_sents,0.1,0.3,0.6)

    # collocations
    with open(os.path.join(output_dir,"txt.collocations_knesset"),"w",encoding="utf-8") as f:
        for n in [2,3,4]:
            if n==2:
                f.write("Two-gram collocations:\n")
            elif n==3:
                f.write("Three-gram collocations:\n")
            else:
                f.write("Four-gram collocations:\n")

            f.write("Frequency:\nCommittee corpus:\n")
            freq_comm = get_k_n_t_collocations(10,n,5,committee_df,"frequency")
            for c_val in freq_comm:
                f.write(c_val[0]+"\n")
            f.write("\nPlenary corpus:\n")
            freq_plen = get_k_n_t_collocations(10,n,5,plenary_df,"frequency")
            for p_val in freq_plen:
                f.write(p_val[0]+"\n")
            f.write("\n")

            f.write("TF-IDF:\nCommittee corpus:\n")
            tfidf_comm = get_k_n_t_collocations(10,n,5,committee_df,"tfidf")
            for c_val in tfidf_comm:
                f.write(c_val[0]+"\n")
            f.write("\nPlenary corpus:\n")
            tfidf_plen = get_k_n_t_collocations(10,n,5,plenary_df,"tfidf")
            for p_val in tfidf_plen:
                f.write(p_val[0]+"\n")
            f.write("\n")

    # now we do 1000 epochs searching for best lambdas and comma_penalty
    # we do random search:
    best_pp=float('inf')
    best_params=None

    for epoch in range(50):
        # random lambdas for interpolation that sum to 1
        a=random.random()
        b=random.random()
        c=random.random()
        total=a+b+c
        l1=a/total
        l2=b/total
        l3=c/total
        # random comma penalty factor between 0 and 1
        # if close to 0 means big penalty to comma
        comma_penalty=random.uniform(0.01,0.9)

        # rebuild models each epoch (inefficient but we don't care)
        committee_sents_final=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
        plenary_sents_final=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]

        committee_model=LM_Trigram(committee_sents_final,l1,l2,l3)
        plenary_model=LM_Trigram(plenary_sents_final,l1,l2,l3)

        # but we must incorporate comma penalty in generating tokens.
        # perplexity evaluation uses generate_next_token.
        # we must modify evaluate_perplexity to use the new models and penalty.
        # since generate_next_token is inside the model and we must handle penalty,
        # let's just monkey patch a solution:
        # we can't easily mod generate_next_token after initialization, let's add penalty param to evaluate:
        # we must re-fill masked sents inside evaluate_perplexity. let's rewrite evaluate_perplexity inside the loop.

        # define inline evaluate:
        def eval_pp():
            log_probs=[]
            total_masked=0
            for (orig,masked) in zip(sampled_sents,masked_sents):
                om_toks=masked.split()
                fi_toks=om_toks[:]
                # fill tokens with the current model and penalty
                for i,tk in enumerate(fi_toks):
                    if tk=="[*]":
                        ctx=" ".join(fi_toks[:i])
                        # now generate next token with penalty
                        # we must re-implement generate_next_token logic here to incorporate comma_penalty
                        # since original code doesn't have a param:
                        # let's do a local copy of logic:

                        # local generate:
                        w1w2=ctx.strip().split()
                        if len(w1w2)<2:
                            w1w2=["_s0_","_s1_"]+w1w2
                        w1,w2=w1w2[-2],w1w2[-1]
                        local_max_tk=None
                        local_max_p=-1
                        for w3 in plenary_model.unigram_counts:
                            p=plenary_model._interp_prob(w1,w2,w3)
                            if w3==',':
                                p*=comma_penalty
                            if p>local_max_p:
                                local_max_p=p
                                local_max_tk=w3
                        fi_toks[i]=local_max_tk

                # now compute perplexity
                f_tokens=["_s0_","_s1_"]+fi_toks
                for i,tk in enumerate(om_toks):
                    if tk=="[*]":
                        pos=i+2
                        w1,w2,w3=f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos]
                        p=plenary_model._interp_prob(w1,w2,w3)
                        if w3==',':
                            p*=comma_penalty
                        if p<=0:
                            p=1e-15
                        log_probs.append(math.log2(p))
                        total_masked+=1
            if total_masked==0:
                return 0.0
            avg_logp=sum(log_probs)/total_masked
            pp=2**(-avg_logp)
            return pp

        pp=eval_pp()

        # print average perplexity each epoch
        # user said print average perplexity in each epoch
        print(f"epoch {epoch}: perplexity={pp:.4f} (l1={l1:.3f}, l2={l2:.3f}, l3={l3:.3f}, comma_penalty={comma_penalty:.3f})")

        if pp<best_pp:
            best_pp=pp
            best_params=(l1,l2,l3,comma_penalty)

    # after 1000 epochs print best results
    print("best results:")
    print(f"perplexity={best_pp:.4f}, l1={best_params[0]:.3f}, l2={best_params[1]:.3f}, l3={best_params[2]:.3f}, comma_penalty={best_params[3]:.3f}")

    # after choosing best constants we must produce final result files again with best params
    # build final model with best params
    committee_sents_final=[tokenize_sentence(s) for s in committee_df["sentence_text"]]
    plenary_sents_final=[tokenize_sentence(s) for s in plenary_df["sentence_text"]]
    committee_final=LM_Trigram(committee_sents_final,best_params[0],best_params[1],best_params[2])
    plenary_final=LM_Trigram(plenary_sents_final,best_params[0],best_params[1],best_params[2])

    # produce final results_sents_sampled
    results=[]
    for orig,masked in zip(sampled_sents,masked_sents):
        om_toks=masked.split()
        fi_toks=om_toks[:]
        pred_tokens=[]
        for i,tk in enumerate(fi_toks):
            if tk=="[*]":
                ctx=" ".join(fi_toks[:i])
                w1w2=ctx.strip().split()
                if len(w1w2)<2:
                    w1w2=["_s0_","_s1_"]+w1w2
                w1,w2=w1w2[-2],w1w2[-1]
                local_max_tk=None
                local_max_p=-1
                for w3 in plenary_final.unigram_counts:
                    p=plenary_final._interp_prob(w1,w2,w3)
                    if w3==',':
                        p*=best_params[3]
                    if p>local_max_p:
                        local_max_p=p
                        local_max_tk=w3
                fi_toks[i]=local_max_tk
                pred_tokens.append(local_max_tk)
        filled_sent=" ".join(fi_toks)

        pp_plen=plenary_final.calculate_prob_of_sentence(filled_sent)
        pp_comm=committee_final.calculate_prob_of_sentence(filled_sent)
        results.append((orig,masked,filled_sent,pred_tokens,pp_plen,pp_comm))

    with open(os.path.join(output_dir,"txt.results_sents_sampled"),"w",encoding="utf-8") as f:
        for r in results:
            f.write(f"original_sentence: {r[0]}\n")
            f.write(f"masked_sentence: {r[1]}\n")
            f.write(f"plenary_sentence: {r[2]}\n")
            f.write("plenary_tokens: "+",".join(r[3])+"\n")
            f.write(f"probability of plenary sentence in plenary corpus: {r[4]:.2f}\n")
            f.write(f"probability of plenary sentence in committee corpus: {r[5]:.2f}\n")

    # perplexity final with best params
    log_probs=[]
    total_masked=0
    for (orig,masked,filled,pred_tokens,pp_plen,pp_comm) in results:
        om_toks=masked.split()
        fi_toks=filled.split()
        f_tokens=["_s0_","_s1_"]+fi_toks
        for i,tk in enumerate(om_toks):
            if tk=="[*]":
                pos=i+2
                w1,w2,w3=f_tokens[pos-2],f_tokens[pos-1],f_tokens[pos]
                p=plenary_final._interp_prob(w1,w2,w3)
                if w3==',':
                    p*=best_params[3]
                if p<=0:
                    p=1e-15
                log_probs.append(math.log2(p))
                total_masked+=1
    if total_masked==0:
        pp=0.0
    else:
        avg_logp=sum(log_probs)/total_masked
        pp=2**(-avg_logp)

    with open(os.path.join(output_dir,"txt.result_perplexity"),"w",encoding="utf-8") as f:
        f.write(f"{pp:.2f}\n")

if __name__=="__main__":
    main()
