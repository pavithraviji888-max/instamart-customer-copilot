import re, csv
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

def make_chunks(overlap=10):
    chunks = []
    for path in sorted(Path("data/policies").glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        step = 40 - overlap
        start = 0
        while start < len(words):
            part = words[start:start+40]
            chunks.append({"source":path.name,"text":" ".join(part)})
            if start + 40 >= len(words):
                break
            start += step
    return chunks

def build_retriever(overlap=10):
    chunks = make_chunks(overlap)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    def retrieve(query,k=3):
        q = vectorizer.transform([query])
        scores = cosine_similarity(q,matrix).ravel()
        ix = scores.argsort()[::-1][:k]
        return [{"source":chunks[i]["source"],"score":float(scores[i]),"text":chunks[i]["text"]} for i in ix]
    return chunks, retrieve

CHUNKS, retrieve = build_retriever(10)

def answer(question, retriever=retrieve):
    found = retriever(question,3)
    context = "\n".join(f"[{x['source']}]\n{x['text']}" for x in found)
    messages = [
        {"role":"system","content":"POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say: I could not find that information in the policy documents."},
        {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
    ]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, [x["source"] for x in found]

def evaluate():
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows=[]
    for _,r in qa.iterrows():
        ans,sources=answer(r.question)
        expected_doc_ok = (r.expected_doc=="NONE") or (r.expected_doc in sources)
        fact_ok = (r.expected_doc=="NONE") or (r.expected_fact in ans)
        fallback = ans.startswith("I could not find")
        rows.append({"question":r.question,"expected_doc":r.expected_doc,"answer":ans,
                     "sources":";".join(sources),"expected_doc_in_top3":expected_doc_ok,
                     "expected_fact_found":fact_ok,"fallback":fallback})
    pd.DataFrame(rows).to_csv("outputs/policy_eval_results.csv",index=False)
    # Zero-overlap evaluation, independent retriever.
    _, retr0 = build_retriever(0)
    rows0=[]
    for _,r in qa.iterrows():
        ans,sources=answer(r.question,retr0)
        rows0.append({"question":r.question,"expected_doc":r.expected_doc,"answer":ans,
                      "sources":";".join(sources),"expected_doc_in_top3":(r.expected_doc=="NONE" or r.expected_doc in sources),
                      "expected_fact_found":(r.expected_doc=="NONE" or r.expected_fact in ans),
                      "fallback":ans.startswith("I could not find")})
    pd.DataFrame(rows0).to_csv("outputs/policy_eval_zero_overlap.csv",index=False)
    print("Policy evaluation with overlap saved; zero-overlap evaluation saved.")

if __name__ == "__main__":
    evaluate()