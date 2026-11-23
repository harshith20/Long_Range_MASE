"""E19 tests 2 + 3 (local): answer vs gold association on the subject page; type / role match of the answer."""
import collections, json, re, sqlite3, zlib
import numpy as np
from hallu_hierarchy.correctness.matching import load_head_words
from hallu_hierarchy.data.normalize import normalize_answer
from hallu_hierarchy.eval.metrics import auroc
from hallu_hierarchy.labeling.cooc import answer_candidates
from hallu_hierarchy.utils import read_jsonl, resolve_pkg_path

R = "hallu_hierarchy/outputs/qwen2.5-3b_popqa_full/"
hw = load_head_words(resolve_pkg_path("configs/head_words.yaml"))
db = sqlite3.connect("hallu_hierarchy/outputs/wiki_cache/wiki.sqlite")
lab = {r["qid"]: r for r in read_jsonl(R + "correctness/labels.jsonl")}
rows = [r for r in read_jsonl(R + "labeling/labels_m3.jsonl") if r.get("label_m3") in ("FA", "AH", "UH")]

def page(title):
    r = db.execute("select rec from pages where title=?", (title,)).fetchone()
    return json.loads(zlib.decompress(r[0])).get("text", "") if r else ""

def count(text, cands):
    return max([len(re.findall(r"(?<!\w)" + re.escape(c) + r"(?!\w)", text)) for c in cands] or [0])

def within(score, cls, rel, neg, pos, mpc=30):
    a = []
    for r in sorted(set(rel)):
        k = (rel == r) & np.isin(cls, [neg, pos]) & ~np.isnan(score)
        y = cls[k] == pos
        if y.sum() >= mpc and (~y).sum() >= mpc:
            a.append(auroc(score[k], y))
    return (np.mean(a) if a else float("nan")), len(a)

# ---------------- test 2: association strength on the subject page
ans_c, gold_c, cls, rel, grp = [], [], [], [], []
for r in rows:
    t = page(r.get("wiki_canonical") or r.get("s_wiki_title") or "")
    if not t:
        continue
    l = lab[r["qid"]]
    a = answer_candidates(l["pred_norm"], hw)
    g = [x for x in dict.fromkeys(normalize_answer(al) for al in l["aliases"]) if len(x) >= 2]
    ans_c.append(count(t, a)); gold_c.append(count(t, g))
    cls.append(r["label_m3"]); rel.append(r["relation"]); grp.append(r["relation_group"])
ans_c, gold_c, cls, rel, grp = map(np.array, (ans_c, gold_c, cls, rel, grp))
print("== TEST 2: mentions on the subject's Wikipedia page (rows with a page:", len(cls), ")")
for c in ("FA", "AH", "UH"):
    m = cls == c
    print(f"  {c}: median mentions of model answer {np.median(ans_c[m]):.0f} | of GOLD {np.median(gold_c[m]):.0f} | "
          f"gold not on page {np.mean(gold_c[m] == 0):.2f}" + (f" | answer mentioned MORE than gold {np.mean(ans_c[m] > gold_c[m]):.2f}" if c == "AH" else ""))
gc = gold_c.astype(float)
print(f"  gold mentions as a score (fewer = more likely wrong): FA vs AH within {within(-gc, cls, rel, 'FA', 'AH')[0]:.3f} | "
      f"FA vs UH {within(-gc, cls, rel, 'FA', 'UH')[0]:.3f}")
ah = cls == "AH"
print(f"  AH: ratio answer/gold mentions median {np.median(ans_c[ah] / np.maximum(gold_c[ah], 1)):.2f}; "
      f"open {np.mean(ans_c[ah & (grp == 'open')] > gold_c[ah & (grp == 'open')]):.2f} vs closed "
      f"{np.mean(ans_c[ah & (grp == 'closed')] > gold_c[ah & (grp == 'closed')]):.2f} answer > gold")

# ---------------- test 3: is the answer a valid filler of the relation, or of another relation (role confusion)?
fillers = collections.defaultdict(set)
for l in lab.values():
    for al in l["aliases"]:
        n = normalize_answer(al)
        if len(n) >= 2:
            fillers[l["relation"]].add(n)
print("== TEST 3: answer type / role")
conf = collections.Counter()
for c in ("FA", "AH", "UH"):
    same = other = 0; n = 0
    for r in rows:
        if r["label_m3"] != c:
            continue
        p = lab[r["qid"]]["pred_norm"]; n += 1
        s = p in fillers[r["relation"]]
        o = [rr for rr, f in fillers.items() if rr != r["relation"] and p in f]
        same += s; other += bool(o) and not s
        if c == "AH" and o and not s:
            conf[(r["relation"], o[0])] += 1
    print(f"  {c}: answer is a gold value of the SAME relation elsewhere {same / n:.2f} | only of ANOTHER relation (role swap) {other / n:.2f} | n={n}")
print("  top AH role swaps (asked relation -> relation the answer is a gold value of):", conf.most_common(8))
