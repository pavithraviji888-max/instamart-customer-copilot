import json, sqlite3, time, uuid
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import mock_llm
from src.part_d_rag import retrieve

Path("state").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
LEDGER = Path("outputs/refund_ledger.jsonl")
TOOLS_SPEC = [
 {"name":"get_order","description":"Look up a clean order in SQLite.","parameters":{"order_id":"string"}},
 {"name":"search_policy","description":"Retrieve policy chunks for a query.","parameters":{"query":"string"}},
 {"name":"calculate_refund","description":"Calculate the eligible late-delivery refund.","parameters":{"order_amount":"number","delivery_minutes":"integer"}},
 {"name":"issue_refund","description":"Append an automatically issued refund to the ledger.","parameters":{"order_id":"string","refund_inr":"number"}},
 {"name":"escalate_to_human","description":"Record a refund requiring human approval.","parameters":{"order_id":"string","refund_inr":"number"}}
]
def get_order(order_id):
    with sqlite3.connect("data/instamart.db") as con:
        row=con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
    if not row: return {"error":"order not found"}
    return {"order_id":row[0],"status":row[1],"order_amount":float(row[2]),"delivery_minutes":None if row[3] is None else int(float(row[3]))}
def search_policy(query):
    return {"results":retrieve(query,3)}
def calculate_refund(order_amount, delivery_minutes):
    amount = 0.0

    if delivery_minutes is not None and int(delivery_minutes) > 45:
        amount = max(0.0, float(order_amount))

    amount = round(amount, 2)

    if amount > 1000:
        approval = True
    else:
        approval = False

    return {
        "refund_inr": amount,
        "requires_human_approval": approval
    }
def _ledger(name,order_id,refund_inr):
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"tool":name,"order_id":order_id,"refund_inr":round(float(refund_inr),2)}
    with LEDGER.open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
    return {"refund_inr":round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return _ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}

def run_agent(user_message,scenario_id="manual",max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tool=None):
    run_id=run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    messages=[{"role":"system","content":"AGENT: Use tools to follow refund policy. Only completed orders are eligible; over INR 1000 must be escalated."},
              {"role":"user","content":user_message}]
    tokens=0; steps=[]; tools_executed=[]; start=time.monotonic(); calls=0
    def save(step_no):
        payload={"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,
                 "step_number":step_no,"tools_executed":tools_executed}
        Path("state",run_id+".json").write_text(json.dumps(payload),encoding="utf-8")
    while calls < max_iterations:
        if time.monotonic()-start >= time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"Human handoff: agent time limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        response=mock_llm.chat(messages)
        calls+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit:
            return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"Human handoff: token limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or "Human handoff required."
            return {"status":content.split(":",1)[0],"final_answer":content,"steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        tc=msg.tool_calls[0]
        args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
        if tc.name == "issue_refund" and float(args.get("refund_inr", 0)) > 1000:
            result = TOOL_MAP["escalate_to_human"](
                order_id=args["order_id"],
                refund_inr=args["refund_inr"]
            )
            tc.name = "escalate_to_human"
        else:
            result = TOOL_MAP[tc.name](**args)
        step={"tool":tc.name,"arguments":args,"observation":result}
        steps.append(step); tools_executed.append(tc.name)
        messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(result)})
        save(len(steps))
        if crash_after_tool is not None and len(tools_executed)==crash_after_tool:
            raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"Human handoff: maximum model-call iterations reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    cp=json.loads(Path("state",run_id+".json").read_text(encoding="utf-8"))
    # Resume the saved conversation without re-running any completed tool.
    messages=cp["messages"]; tokens=cp["tokens_used"]; steps=[]
    for m in messages:
        if m.get("role")=="tool":
            try: obs=json.loads(m["content"])
            except Exception: obs={}
            steps.append({"tool":m.get("name"),"arguments":{},"observation":obs})
    calls=0; start=time.monotonic()
    while calls < max_iterations:
        if time.monotonic()-start >= time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"Human handoff: agent time limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit:
            return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"Human handoff: token limit reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or "Human handoff required."
            return {"status":content.split(":",1)[0],"final_answer":content,"steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}
        tc=msg.tool_calls[0]; args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
        result=TOOL_MAP[tc.name](**args)
        steps.append({"tool":tc.name,"arguments":args,"observation":result})
        cp["tools_executed"].append(tc.name); tokens_now=tokens
        messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(result)})
        cp.update({"messages":messages,"tokens_used":tokens_now,"step_number":len(cp["tools_executed"])})
        Path("state",run_id+".json").write_text(json.dumps(cp),encoding="utf-8")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"Human handoff: maximum model-call iterations reached.","steps":steps,"llm_calls":calls,"tokens_used":tokens,"run_id":run_id}

def run_part_d():
    scenarios=pd.read_csv("data/raw/agent_scenarios.csv",keep_default_na=False)
    LEDGER.write_text("",encoding="utf-8")
    out=[]
    for _,r in scenarios.iterrows():
        result=run_agent(r.message,r.scenario_id)
        record={"scenario_id":r.scenario_id,**result}
        out.append(record)
    with open("outputs/agent_runs.jsonl","w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r)+"\n")
    print("Agent scenario runs saved.")

if __name__ == "__main__":
    run_part_d()