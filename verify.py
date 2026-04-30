import json

r = json.load(open("results/visual_cot_verification_results.json"))
n = len(r)
flipped = sum(1 for x in r if x["answer_flipped"])
p1_correct_but_flipped = sum(
    1 for x in r if x["pass1_answer_correct"] and x["answer_flipped"]
)
recovered = sum(1 for x in r if x["recovered"])
print(f"Flipped: {flipped}/{n}")
print(f"  of which were correct in P1: {p1_correct_but_flipped}")
print(f"Recovered (wrong→right): {recovered}")

