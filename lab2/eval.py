import json, os
from pathlib import Path
from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError
from common.call_llm import call_llm
from common.prompts import system_prompt
from common.tool_schema import TOOLS
from common.utils import extract_json_block

TOOL_INDEX = {t["name"]: t for t in TOOLS}

def validate_tool_call(tool_call):
    if not isinstance(tool_call, dict) or tool_call.get("type") != "tool_call":
        return False, "invalid_format"
    name = tool_call.get("name")
    if name not in TOOL_INDEX: return False, f"unknown_tool:{name}"
    try:
        jsonschema_validate(instance=tool_call.get("arguments"), schema=TOOL_INDEX[name]["parameters"])
        return True, None
    except ValidationError as e: return False, str(e)

def tool_selection_correct(pred_tool, expect):
    if "tool" in expect: return pred_tool == expect["tool"]
    return pred_tool is None

def args_exact_match(pred_args, expect):
    if "arguments" not in expect: return pred_args is None
    return pred_args == expect["arguments"]

def run_one(case):
    messages = [{"role": "system", "content": system_prompt()}] + case["messages"]
    out = call_llm(messages)
    pred = {"raw": out, "tool": None, "arguments": None, "valid": False, "error": None}
    try:
        tc = extract_json_block(out)
        if tc:
            ok, err = validate_tool_call(tc)
            pred.update({"tool": tc.get("name"), "arguments": tc.get("arguments"), "valid": ok, "error": err})
    except Exception as e: pred["error"] = str(e)
    return pred

def main():
    print("=" * 60 + "\nLAB2: 評估開始\n" + "=" * 60)
    with open(Path(__file__).parent / "eval_cases.json", encoding="utf-8") as f:
        cases = json.load(f)
    rows = []
    for c in cases:
        pred = run_one(c)
        rows.append({"valid": pred["valid"], "tool_ok": tool_selection_correct(pred["tool"], c["expect"]), 
                     "args_ok": args_exact_match(pred["arguments"], c["expect"]) if "arguments" in c["expect"] else True})
    n = len(rows)
    print(f"總數: {n}")
    print(f"格式合法率: {sum(1 for r in rows if r['valid'])/n:.1%}")
    print(f"工具準確率: {sum(1 for r in rows if r['tool_ok'])/n:.1%}")
    print(f"參數相符率: {sum(1 for r in rows if r['args_ok'])/n:.1%}")

if __name__ == "__main__": main()
