import json, re, random, time
from pathlib import Path
import pandas as pd
import mock_llm

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\b[6-9]\d{9}\b")
ORDER = re.compile(r"ORD-\d{6}")
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = {"billing","delivery","technical","fraud"}

ZERO_PROMPT = """CLASSIFY each support ticket. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}."""
FEW_PROMPT = ZERO_PROMPT + """
EXAMPLES:
Ticket: Unauthorized transaction; I did not make this payment.
{"category":"fraud","reason":"unauthorized transaction"}
Ticket: The app crashes when I open the cart.
{"category":"technical","reason":"app crash"}
Ticket: I was charged twice for one order.
{"category":"billing","reason":"duplicate charge"}
"""

def redact(text):
    counts = {}
    text, counts["emails"] = EMAIL.subn("[EMAIL]", text)
    text, counts["phones"] = PHONE.subn("[PHONE]", text)
    text, counts["order_ids"] = ORDER.subn("[ORDER_ID]", text)
    assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text)
    return text, counts

def classify_batch(prompt, budget=100000, use_mock_reset=True):
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    segments = pd.read_csv("outputs/customer_segments.csv")
    segmap = segments.set_index("customer_id")["segment_name"].to_dict()
    results, running_tokens = [], 0
    if use_mock_reset:
        mock_llm.reset()
    for _, row in tickets.iterrows():
        tid, raw = row.ticket_id, row.text
        status, category, tokens = "", "", 0
        if not raw.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(raw) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif running_tokens >= budget:
            status = "budget_exhausted"
        else:
            redacted, _ = redact(raw)
            messages = [{"role":"system","content":prompt},{"role":"user","content":redacted}]
            attempts = 0
            while True:
                try:
                    response = mock_llm.chat(messages)
                    break
                except mock_llm.TransientLLMError:
                    if attempts >= 3:
                        status = "flagged_invalid_output"
                        response = None
                        break
                    wait = BASE_DELAY * (2 ** attempts) * random.uniform(0.9,1.15)
                    print(f"retry ticket={tid} retry={attempts} wait={wait:.4f}s")
                    time.sleep(wait)
                    attempts += 1
            if response is not None:
                tokens = response.usage.total_tokens
                running_tokens += tokens
                content = response.choices[0].message.content or ""
                try:
                    obj = json.loads(content)
                    if obj.get("category") in ALLOWED:
                        category, status = obj["category"], "classified_json"
                    else:
                        status = "flagged_invalid_output"
                except (json.JSONDecodeError, TypeError):
                    words = re.findall(r"[a-z]+", content.lower())
                    found = [w for w in words if w in ALLOWED]
                    if len(set(found)) == 1:
                        category, status = found[0], "classified_keyword"
                    else:
                        status = "flagged_invalid_output"
        if status in {"skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"}:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif bool(segments.loc[segments.customer_id.eq(row.customer_id),"is_premium"].any()):
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        results.append({"ticket_id":tid,"customer_id":row.customer_id,"status":status,"category":category,
                        "segment_name":segmap.get(row.customer_id,""),"priority":priority,"route":route,"tokens_used":tokens,
                        "true_category":row.true_category})
    return pd.DataFrame(results)

def run_part_c():
    zero = classify_batch(ZERO_PROMPT, 100000)
    eligible = zero[zero.status.isin(["classified_json","classified_keyword"])]
    print("Zero-shot classified:",len(eligible),"accuracy:",(eligible.category==eligible.true_category).mean())
    zero.to_csv("outputs/triage_zero_shot.csv",index=False)
    budget = classify_batch(ZERO_PROMPT, 5000)
    budget.to_csv("outputs/triage_budget_demo.csv",index=False)
    few = classify_batch(FEW_PROMPT,100000)
    few.to_csv("outputs/triage_results.csv",index=False)
    print("Few-shot classified:",int(few.status.isin(["classified_json","classified_keyword"]).sum()))
    print("Status counts:\n",few.status.value_counts())

if __name__ == "__main__":
    run_part_c()