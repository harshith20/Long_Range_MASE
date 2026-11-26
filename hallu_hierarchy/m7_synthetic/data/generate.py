"""M7 / E20: synthetic biography corpus with controlled exposure counts (plan §1).

Usage (from the repo root):
    python -m hallu_hierarchy.m7_synthetic.data.generate --config hallu_hierarchy/m7_synthetic/data/config_v1.yaml \
        --budget-only
    python -m hallu_hierarchy.m7_synthetic.data.generate --config ... --fraction 0.01 \
        --out hallu_hierarchy/outputs/m7_synthetic/dryrun_1pct

Steps: build all people of the full design -> plan exposure counts -> print the token budget (nothing written
yet) -> select the people of this run (all of them, or a fraction per cell) -> render documents into random
buckets on disk -> train BPE -> write the corpus in training order with an exposure log.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .lexicon import make_lexicon, random_date

HERE = Path(__file__).resolve().parent
ATTRS = ["birth_date", "birth_city", "university", "major", "employer", "work_city"]
OTHER_ATTRS = ["birth_date", "university", "major", "employer"]
CTX = ["bio", "standalone", "news", "table", "cross", "qa", "filler"]
EOS = "<|endoftext|>"


# ----------------------------------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_templates(path: str | Path = HERE / "templates.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------- people
def build_people(cfg: dict) -> tuple[list[dict], dict]:
    """All people of the full design, with attributes and factor levels (no partners yet)."""
    seed = cfg["seed"]
    lx_cfg = cfg["lexicon"]
    lex = make_lexicon(seed, lx_cfg["n_cities"], lx_cfg["n_universities"], lx_cfg["n_employers"],
                       lx_cfg["n_first"], lx_cfg["n_middle"], lx_cfg["n_last"])
    fac = cfg["factors"]
    rng = random.Random(seed + 1)
    cities = lex["cities"]
    n_per = fac["people_per_cell_split"]
    used_names: set[str] = set()
    people: list[dict] = []
    for e in fac["exp_birth"]:
        for r in fac["ratio"]:
            if fac.get("drop_over_cap") and e * r > fac["work_cap"]:
                continue  # plan §1.2 cap rule: drop the cell instead of capping its ratio
            for fam, fam_n in fac["familiarity"].items():
                for split in ("qa_train", "qa_test"):
                    for k in range(n_per):
                        while True:
                            name = f"{rng.choice(lex['first'])} {rng.choice(lex['middle'])} {rng.choice(lex['last'])}"
                            if name not in used_names:
                                used_names.add(name)
                                break
                        work_n = min(e * r, fac["work_cap"])
                        if fac.get("drop_over_cap"):
                            assert work_n == e * r, "actual ratio must equal the labelled ratio"
                        people.append({
                            "person_id": len(people), "name": name, "cell_rank": k,
                            "birth_date": random_date(rng), "major": rng.choice(lex["majors"]),
                            "exp_birth": e, "ratio": r, "familiarity": fam, "split": split,
                            "n_birth_date": fam_n, "n_birth_city": e, "n_university": fam_n,
                            "n_major": fam_n, "n_employer": fam_n, "n_work_city": work_n,
                            "eff_ratio": work_n / e, "capped": work_n < e * r,
                        })
    org_w = org_city_weight(load_templates())
    assign_cities(people, lex, rng, cfg.get("city_balance", False), org_w)
    people_meta = {"n_cities": len(cities), "cities": cities, "org_city_weight": org_w, "lex": lex,
                   "city_load": city_load(people, cities, org_w)}
    return people, people_meta


def org_city_weight(tpl: dict) -> dict[str, float]:
    """Share of university / employer statement templates that also name the organisation's city."""
    st = tpl["statements"]
    return {"university": sum("{ucity}" in t for t in st["university"]["train"]) / len(st["university"]["train"]),
            "employer": sum("{ecity}" in t for t in st["employer"]["train"]) / len(st["employer"]["train"])}


def assign_cities(people: list[dict], lex: dict, rng: random.Random, balance: bool, org_w: dict) -> None:
    """Birth city, work city, university and employer per person; all four cities differ (D4).

    balance=False: uniform at random (gen-v0.1).
    balance=True (E20b): heavy people (exp 300 x ratio 30 = 3,000 work-city sentences; high familiarity = 100
    university/employer sentences) would otherwise make some cities far more frequent than others. A model that
    does not know a fact tends to guess frequent cities, so those guesses would look like AH. Greedy fill, heaviest
    person first: each of birth city, work city, university, employer goes to the least-loaded allowed city.
    Load = expected sentences naming the city (org sentences weighted by org_w). Ties broken at random.
    """
    cities = lex["cities"]
    if not balance:
        for p in people:
            birth = rng.choice(cities)
            work = rng.choice([c for c in cities if c != birth])
            uni = rng.choice([u for u in lex["universities"] if lex["university_city"][u] not in (birth, work)])
            ucity = lex["university_city"][uni]
            emp = rng.choice([x for x in lex["employers"] if lex["employer_city"][x] not in (birth, work, ucity)])
            p.update({"birth_city": birth, "work_city": work, "university": uni, "university_city": ucity,
                      "employer": emp, "employer_city": lex["employer_city"][emp]})
        return
    load = {c: 0.0 for c in cities}
    tie = {c: rng.random() for c in cities}
    key = lambda c: (load[c], tie[c])
    unis, emps = list(lex["universities"]), list(lex["employers"])
    rng.shuffle(unis)
    rng.shuffle(emps)
    weight = lambda p: p["n_birth_city"] + p["n_work_city"] + org_w["university"] * p["n_university"] \
        + org_w["employer"] * p["n_employer"]
    for p in sorted(people, key=lambda p: (-weight(p), rng.random())):
        birth = min(cities, key=key)
        load[birth] += p["n_birth_city"]
        work = min((c for c in cities if c != birth), key=key)
        load[work] += p["n_work_city"]
        uni = min((u for u in unis if lex["university_city"][u] not in (birth, work)),
                  key=lambda u: key(lex["university_city"][u]))
        ucity = lex["university_city"][uni]
        load[ucity] += org_w["university"] * p["n_university"]
        emp = min((x for x in emps if lex["employer_city"][x] not in (birth, work, ucity)),
                  key=lambda x: key(lex["employer_city"][x]))
        load[lex["employer_city"][emp]] += org_w["employer"] * p["n_employer"]
        p.update({"birth_city": birth, "work_city": work, "university": uni, "university_city": ucity,
                  "employer": emp, "employer_city": lex["employer_city"][emp]})


def city_load(people: list[dict], cities: list[str], org_w: dict) -> dict[str, int]:
    """Expected sentences naming each city: birth + work sentences, plus university / employer sentences times
    the share of their templates that name the organisation's city."""
    load = {c: 0.0 for c in cities}
    for p in people:
        load[p["birth_city"]] += p["n_birth_city"]
        load[p["work_city"]] += p["n_work_city"]
        load[p["university_city"]] += org_w["university"] * p["n_university"]
        load[p["employer_city"]] += org_w["employer"] * p["n_employer"]
    return {c: round(v) for c, v in load.items()}


def select_people(people: list[dict], fraction: float, n_per: int) -> tuple[list[dict], int]:
    """First k people of every (cell, split); k = max(1, round(n_per * fraction))."""
    if fraction >= 1.0:
        return [dict(p) for p in people], n_per
    k = max(1, round(n_per * fraction))
    return [dict(p) for p in people if p["cell_rank"] < k], k


def spread(load: dict[str, int]) -> dict:
    v = np.array(list(load.values()), dtype=float)
    return {"min": int(v.min()), "median": int(np.median(v)), "max": int(v.max()),
            "max_over_median": round(float(v.max() / max(1.0, np.median(v))), 2),
            "cv": round(float(v.std() / max(1.0, v.mean())), 3)}


def print_city_spread(title: str, load: dict[str, int]) -> None:
    s = spread(load)
    print(f"  {title}: min {s['min']:,} | median {s['median']:,} | max {s['max']:,} "
          f"(max/median {s['max_over_median']}, cv {s['cv']})")


def add_freq_chance(people: list[dict], load: dict[str, int]) -> None:
    """Chance-level AH if a model that does not know the fact guesses cities in proportion to their corpus
    frequency: mentions of the linked cities / mentions of all cities except the true birth city (E20b)."""
    total = sum(load.values())
    for p in people:
        rest = total - load[p["birth_city"]]
        p["chance_ah_freq"] = sum(load[c] for c in p["linked_cities"]) / max(1, rest)


# ----------------------------------------------------------------------------------------------- planning
def plan(people: list[dict], cfg: dict, n_cities: int) -> None:
    """Partners, cross-document allocations, name-balance targets. Mutates people in place."""
    rng = random.Random(cfg["seed"] + 2)
    co = cfg["cooccurrence"]
    fac = cfg["factors"]
    ids = [p["person_id"] for p in people]
    by_id = {p["person_id"]: p for p in people}
    partners: dict[int, set[int]] = defaultdict(set)
    for _ in range(co["max_partners"]):
        perm = ids[:]
        rng.shuffle(perm)
        for a, b in zip(perm[0::2], perm[1::2]):
            if len(partners[a]) < co["max_partners"] and len(partners[b]) < co["max_partners"] and a != b:
                partners[a].add(b)
                partners[b].add(a)
    anchor_mentions: Counter = Counter()
    def r_max(e: int) -> int:  # largest ratio that exists for this exp_birth level
        ok = [r for r in fac["ratio"] if not (fac.get("drop_over_cap") and e * r > fac["work_cap"])]
        return max(ok)
    for p in people:
        pid = p["person_id"]
        p["partners"] = sorted(partners[pid])
        counts = {a: p[f"n_{a}"] for a in ATTRS}
        total = sum(counts.values())
        n_cross = min(co["cross_max_per_person"], round(co["cross_fraction"] * total)) if p["partners"] else 0
        cross = []
        prng = random.Random(cfg["seed"] * 1_000_003 + pid)
        for _ in range(n_cross):
            attrs = [a for a in ATTRS if counts[a] > 0]
            attr = prng.choices(attrs, weights=[counts[a] for a in attrs])[0]
            counts[attr] -= 1
            anchor = prng.choice(p["partners"])
            anchor_mentions[anchor] += 1
            cross.append((anchor, attr))
        p["cross"] = cross
        fam_n = p["n_birth_date"]
        e = p["exp_birth"]
        p["name_target"] = e + min(r_max(e) * e, fac["work_cap"]) + len(OTHER_ATTRS) * fam_n
    for p in people:
        p["anchor_mentions"] = anchor_mentions[p["person_id"]]
        # linked cities = cities that co-occur with this person in text (D4)
        linked = {p["university_city"], p["employer_city"]}
        if p["n_work_city"] > 0:
            linked.add(p["work_city"])
        for anchor, _ in p["cross"]:
            a = by_id[anchor]
            linked.add(a["birth_city"])
            if a["n_work_city"] > 0:
                linked.add(a["work_city"])
        linked.discard(p["birth_city"])
        p["linked_cities"] = sorted(linked)
        p["chance_ah"] = len(linked) / (n_cities - 1)
    # anchors also co-occur with the partner's cities: add in a second pass
    for p in people:
        for anchor, _ in p["cross"]:
            a = by_id[anchor]
            extra = {p["birth_city"]} | ({p["work_city"]} if p["n_work_city"] > 0 else set())
            merged = (set(a["linked_cities"]) | extra) - {a["birth_city"]}
            a["linked_cities"] = sorted(merged)
            a["chance_ah"] = len(merged) / (n_cities - 1)


def planned_counts(p: dict, cfg: dict) -> dict:
    """Sentence counts for one person before rendering (budget estimate)."""
    facts = sum(p[f"n_{a}"] for a in ATTRS)
    filler = max(0, p["name_target"] - facts - p["anchor_mentions"]) if cfg["name_balance"] else 0
    qa = 0
    if p["split"] == "qa_train":
        n_attrs = len(ATTRS) - (1 if p["n_work_city"] == 0 else 0)
        qa = n_attrs * cfg["qa"]["pairs_per_attr"]
    return {"facts": facts, "filler": filler, "qa": qa}


# ----------------------------------------------------------------------------------------------- budget
def _mean_len(strings: list[str]) -> float:
    return sum(len(s) for s in strings) / max(1, len(strings))


def estimate_budget(people: list[dict], cfg: dict, tpl: dict) -> dict:
    """Token estimate from planned counts and average filled-template length (chars / chars_per_token)."""
    cpt = cfg["budget"]["chars_per_token_guess"]
    name_len = _mean_len([p["name"] for p in people[:2000]])
    stmt = [t for a in ATTRS for t in tpl["statements"][a]["train"]]
    fact_chars = _mean_len(stmt) - 6 + name_len + 6  # placeholders ~ value length
    gen_chars = _mean_len(tpl["generic_filler"])
    w = cfg["context_weights"]
    news_extra = w["news"] / sum(w.values()) * (sum(cfg["news_filler"]) / 2) * gen_chars
    fact_doc_chars = fact_chars + news_extra + 2
    filler_chars = _mean_len(tpl["name_only"]) - 6 + name_len + (sum(cfg["name_only_filler"]) / 2) * gen_chars
    qa_chars = _mean_len([q for a in ATTRS for q in tpl["questions"][a]["train"]]) + name_len + 15
    rows = []
    for p in people:
        c = planned_counts(p, cfg)
        tok = (c["facts"] * fact_doc_chars + c["filler"] * filler_chars + c["qa"] * qa_chars) / cpt
        rows.append((p, c, tok))
    return _budget_table(rows)


def _budget_table(rows: list[tuple[dict, dict, float]]) -> dict:
    out = {"n_people": len(rows), "tokens_total": 0.0, "facts": 0, "filler": 0, "qa": 0,
           "tokens_by_exp_birth": Counter(), "tokens_by_ratio": Counter(), "tokens_by_familiarity": Counter(),
           "tokens_by_split": Counter(), "tokens_by_kind": Counter()}
    for p, c, tok in rows:
        out["tokens_total"] += tok
        for k in ("facts", "filler", "qa"):
            out[k] += c[k]
        out["tokens_by_exp_birth"][str(p["exp_birth"])] += tok
        out["tokens_by_ratio"][str(p["ratio"])] += tok
        out["tokens_by_familiarity"][p["familiarity"]] += tok
        out["tokens_by_split"][p["split"]] += tok
    for k in [k for k in out if k.startswith("tokens_by")]:
        out[k] = {kk: round(v) for kk, v in sorted(out[k].items(), key=lambda kv: _num(kv[0]))}
    out["tokens_total"] = round(out["tokens_total"])
    return out


def _num(s: str):
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def print_budget(title: str, b: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"people {b['n_people']:,} | fact sentences {b['facts']:,} | balancing name mentions {b['filler']:,} "
          f"| QA pairs {b['qa']:,}")
    print(f"tokens total ≈ {b['tokens_total'] / 1e6:,.1f} M")
    for key in ("tokens_by_exp_birth", "tokens_by_ratio", "tokens_by_familiarity", "tokens_by_split",
                "tokens_by_kind"):
        if b.get(key):
            parts = " | ".join(f"{k}: {v / 1e6:,.1f}M" for k, v in b[key].items())
            print(f"  {key[10:]:<12} {parts}")


# ----------------------------------------------------------------------------------------------- render
class Renderer:
    def __init__(self, tpl: dict, cfg: dict, by_id: dict[int, dict]):
        self.tpl, self.cfg, self.by_id = tpl, cfg, by_id
        self.stmt = {a: tpl["statements"][a]["train"] for a in ATTRS}
        self.org_city_tpl = {a: [("{ucity}" in t or "{ecity}" in t) for t in self.stmt[a]] for a in ATTRS}

    def fill(self, t: str, p: dict, name: str | None = None) -> str:
        return t.format(name=name or p["name"], date=p["birth_date"], city=p["birth_city"],
                        university=p["university"], ucity=p["university_city"], major=p["major"],
                        employer=p["employer"], ecity=p["employer_city"])

    def fact(self, rng: random.Random, p: dict, attr: str) -> tuple[str, int]:
        i = rng.randrange(len(self.stmt[attr]))
        t = self.stmt[attr][i]
        if attr == "work_city":
            t = t.replace("{city}", "{wcity}")
            s = t.format(name=p["name"], wcity=p["work_city"])
        else:
            s = self.fill(t, p)
        return s, i

    def value(self, p: dict, attr: str) -> str:
        return {"birth_date": p["birth_date"], "birth_city": p["birth_city"], "university": p["university"],
                "major": p["major"], "employer": p["employer"], "work_city": p["work_city"]}[attr]

    def person_docs(self, p: dict) -> tuple[list[dict], int, int, int]:
        """Documents about p (facts, balancing, QA). Returns docs, name mentions, balancing docs, QA pairs."""
        cfg, tpl = self.cfg, self.tpl
        rng = random.Random(cfg["seed"] * 7_000_003 + p["person_id"])
        pid = p["person_id"]
        docs: list[dict] = []
        counts = {a: p[f"n_{a}"] for a in ATTRS}
        mentions = 0
        # 1) cross-person documents (counts for p only; anchor's name co-occurs)
        for anchor_id, attr in p["cross"]:
            counts[attr] -= 1
            anchor = self.by_id[anchor_id]
            intro = rng.choice(tpl["cross_intro"]).format(anchor=anchor["name"], name=p["name"],
                                                         rel=rng.choice(tpl["relations"]))
            s, i = self.fact(rng, p, attr)
            docs.append({"t": f"{intro} {s}", "k": "cross", "e": [[pid, attr, "cross", i]],
                         "c": [anchor_id, pid]})
            mentions += 2  # name in intro and in the fact sentence
        # 2) the remaining exposures, in random context types
        w = cfg["context_weights"]
        kinds, weights = list(w), list(w.values())
        while True:
            avail = [a for a in ATTRS if counts[a] > 0]
            if not avail:
                break
            kind = rng.choices(kinds, weights=weights)[0]
            if kind == "bio" and len(avail) < 2:
                kind = "standalone"
            if kind == "bio":
                k = rng.randint(cfg["bio_attrs"][0], min(cfg["bio_attrs"][1], len(avail)))
                chosen = _weighted_sample(rng, avail, [counts[a] for a in avail], k)
                sents, exps = [], []
                for a in chosen:
                    counts[a] -= 1
                    s, i = self.fact(rng, p, a)
                    sents.append(s)
                    exps.append([pid, a, "bio", i])
                docs.append({"t": " ".join(sents), "k": "fact", "e": exps, "c": None})
                mentions += len(sents)
            elif kind == "table":
                k = rng.randint(cfg["table_attrs"][0], min(cfg["table_attrs"][1], len(avail)))
                chosen = _weighted_sample(rng, avail, [counts[a] for a in avail], k)
                heads, vals, exps = ["Name"], [p["name"]], []
                for a in chosen:
                    counts[a] -= 1
                    heads.append(rng.choice(tpl["table_headers"][a]))
                    vals.append(self.value(p, a))
                    exps.append([pid, a, "table", -1])
                docs.append({"t": " | ".join(heads) + "\n" + " | ".join(vals), "k": "fact", "e": exps, "c": None})
                mentions += 1
            else:  # standalone or news: one fact
                a = rng.choices(avail, weights=[counts[x] for x in avail])[0]
                counts[a] -= 1
                s, i = self.fact(rng, p, a)
                if kind == "news":
                    sents = rng.sample(tpl["generic_filler"], rng.randint(*cfg["news_filler"]))
                    sents.insert(rng.randint(0, len(sents)), s)
                    text = " ".join(sents)
                else:
                    text = s
                docs.append({"t": text, "k": "fact", "e": [[pid, a, kind, i]], "c": None})
                mentions += 1
        # 3) name-mention balancing (D2); anchor mentions from other people's cross docs count too
        n_fill = 0
        if cfg["name_balance"]:
            n_fill = max(0, p["name_target"] - mentions - p["anchor_mentions"])
        for _ in range(n_fill):
            sents = rng.sample(tpl["generic_filler"], rng.randint(*cfg["name_only_filler"]))
            sents.insert(rng.randint(0, len(sents)), rng.choice(tpl["name_only"]).format(name=p["name"]))
            docs.append({"t": " ".join(sents), "k": "filler", "e": [], "c": None})
        mentions += n_fill
        # 4) QA pairs for qa_train people (not exposures; logged with is_exposure = False)
        n_qa = 0
        if p["split"] == "qa_train":
            for a in ATTRS:
                if a == "work_city" and p["n_work_city"] == 0:
                    continue  # would expose the work city at ratio 0
                qs = tpl["questions"][a]["train"]
                # D6 (Allen-Zhu Part 3.1): QA is a large share of training; cycle through the train templates
                order = rng.sample(range(len(qs)), len(qs))
                for j in range(cfg["qa"]["pairs_per_attr"]):
                    qi = order[j % len(qs)]
                    q = qs[qi].format(name=p["name"])
                    docs.append({"t": f"Q: {q}\nA: {self.value(p, a)}", "k": "qa", "e": [[pid, a, "qa", qi]],
                                 "c": None})
                    n_qa += 1
        return docs, mentions, n_fill, n_qa


def _weighted_sample(rng: random.Random, items: list[str], weights: list[int], k: int) -> list[str]:
    items, weights = list(items), list(weights)
    out = []
    for _ in range(k):
        j = rng.choices(range(len(items)), weights=weights)[0]
        out.append(items.pop(j))
        weights.pop(j)
    return out


# ----------------------------------------------------------------------------------------------- tokenizer
def train_tokenizer(people: list[dict], tpl: dict, renderer: Renderer, vocab_size: int, cities: list[str]):
    """BPE on one full biography per person (of the FULL design) + every template, so that a person's
    exposure count does not decide whether their name becomes a single token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    def texts():
        rng = random.Random(0)
        for p in people:
            yield " ".join(renderer.fact(rng, p, a)[0] for a in ATTRS)
        for a in ATTRS:
            for t in tpl["statements"][a]["train"]:
                yield t
            for q in tpl["questions"][a]["train"]:
                yield "Q: " + q + "\nA:"
        for s in tpl["generic_filler"] + tpl["name_only"] + tpl["cross_intro"]:
            yield s
        for a in ATTRS:
            yield " | ".join(tpl["table_headers"][a])
        for _ in range(3):  # every city becomes one token: the answer is decided at a single token
            yield " " + " ".join(cities)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts(), trainer)
    return tok


# ----------------------------------------------------------------------------------------------- write
def generate(cfg: dict, tpl: dict, out: Path, fraction: float, budget_only: bool = False,
             overwrite: bool = False, quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else print
    t0 = time.time()
    all_people, meta = build_people(cfg)
    n_cities = meta["n_cities"]
    full = [dict(p) for p in all_people]
    plan(full, cfg, n_cities)
    est_full = estimate_budget(full, cfg, tpl)
    if not quiet:
        print_budget("BUDGET ESTIMATE, full design (before writing; chars/token guess "
                     f"{cfg['budget']['chars_per_token_guess']})", est_full)
        n_capped = sum(p["capped"] for p in full)
        print(f"  capped cells: {n_capped:,} people do not reach their nominal ratio (D3)")
        print_city_spread("city mentions, full design", meta["city_load"])
    if budget_only:
        return {"estimate_full": est_full}

    people, k = select_people(all_people, fraction, cfg["factors"]["people_per_cell_split"])
    if fraction < 1.0 and cfg.get("city_balance", False):
        # a subset of a balanced design is not balanced (few heavy people dominate): re-assign within the subset
        assign_cities(people, meta["lex"], random.Random(cfg["seed"] + 5), True, meta["org_city_weight"])
    plan(people, cfg, n_cities)  # partners and cross docs within the selected people
    run_load = city_load(people, meta["cities"], meta["org_city_weight"]) if fraction < 1.0 else meta["city_load"]
    add_freq_chance(people, run_load)
    by_id = {p["person_id"]: p for p in people}
    if not quiet:
        print(f"\nrun: {len(people):,} people ({k} per cell × split), fraction {fraction}")

    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; raw outputs are never overwritten (plan §9)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_buckets"
    tmp.mkdir()

    renderer = Renderer(tpl, cfg, by_id)
    n_buckets = cfg["output"]["shuffle_buckets"]
    brng = random.Random(cfg["seed"] + 3)
    bfiles = [open(tmp / f"b{b:03d}.jsonl", "w", encoding="utf-8") for b in range(n_buckets)]
    for p in people:
        docs, mentions, n_fill, n_qa = renderer.person_docs(p)
        p["name_mentions"] = mentions + p["anchor_mentions"]
        p["n_filler"] = n_fill
        p["n_qa"] = n_qa
        for d in docs:
            bfiles[brng.randrange(n_buckets)].write(json.dumps(d, ensure_ascii=False) + "\n")
    for f in bfiles:
        f.close()
    log(f"rendered in {time.time() - t0:.1f} s")

    # tokenizer
    tcfg = cfg["tokenizer"]
    if tcfg.get("path"):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tcfg["path"])
    else:
        full_by_id = {p["person_id"]: p for p in full}
        tok = train_tokenizer(full, tpl, Renderer(tpl, cfg, full_by_id), tcfg["vocab_size"], meta["cities"])
    tok.save(str(out / "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)
    log(f"tokenizer ready ({tok.get_vocab_size()} tokens) at {time.time() - t0:.1f} s")

    # final pass: shuffle each bucket, write corpus + tokens + logs in training order
    import pyarrow as pa
    import pyarrow.parquet as pq

    exp_writer = None
    exp_schema = pa.schema([("person_id", pa.int32()), ("attribute", pa.string()), ("exposure_index", pa.int32()),
                            ("doc_idx", pa.int64()), ("token_offset", pa.int64()), ("context_type", pa.string()),
                            ("template_id", pa.int16()), ("is_exposure", pa.bool_())])
    exp_writer = pq.ParquetWriter(out / "exposure_log.parquet", exp_schema)
    exp_counter: Counter = Counter()
    cooc_rows, doc_rows = [], []
    kind_tokens: Counter = Counter()
    kind_docs: Counter = Counter()
    ctx_docs: Counter = Counter()
    shard_size = cfg["output"]["shard_tokens"]
    shard_id, shard_buf, shard_len = 0, [], 0
    offset = 0
    doc_idx = 0
    srng = random.Random(cfg["seed"] + 4)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as corpus:
        for b in range(n_buckets):
            with open(tmp / f"b{b:03d}.jsonl", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f]
            srng.shuffle(docs)
            encs = tok.encode_batch([d["t"] for d in docs])
            cols = {k: [] for k in exp_schema.names}
            for d, enc in zip(docs, encs):
                ids = enc.ids + [eos_id]
                n = len(ids)
                corpus.write(json.dumps({"doc_idx": doc_idx, "kind": d["k"], "text": d["t"]}, ensure_ascii=False) + "\n")
                shard_buf.append(np.asarray(ids, dtype=np.uint16))
                shard_len += n
                kind = d["k"]
                kind_tokens[kind] += n
                kind_docs[kind] += 1
                ctx_docs[d["e"][0][2] if d["e"] else kind] += 1
                for pid, attr, ctx, tid in d["e"]:
                    key = (pid, attr, ctx == "qa")
                    cols["person_id"].append(pid)
                    cols["attribute"].append(attr)
                    cols["exposure_index"].append(exp_counter[key])
                    cols["doc_idx"].append(doc_idx)
                    cols["token_offset"].append(offset)
                    cols["context_type"].append(ctx)
                    cols["template_id"].append(tid)
                    cols["is_exposure"].append(ctx != "qa")
                    exp_counter[key] += 1
                if d["c"]:
                    cooc_rows.append({"anchor_id": d["c"][0], "person_id": d["c"][1], "doc_idx": doc_idx})
                doc_rows.append((doc_idx, kind, offset, n))
                offset += n
                doc_idx += 1
                if shard_len >= shard_size:
                    np.concatenate(shard_buf).tofile(out / f"tokens_{shard_id:03d}.bin")
                    shard_id, shard_buf, shard_len = shard_id + 1, [], 0
            exp_writer.write_table(pa.table(cols, schema=exp_schema))
    if shard_buf:
        np.concatenate(shard_buf).tofile(out / f"tokens_{shard_id:03d}.bin")
    exp_writer.close()
    shutil.rmtree(tmp)

    import pandas as pd
    pd.DataFrame(cooc_rows, columns=["anchor_id", "person_id", "doc_idx"]).to_parquet(out / "cooccurrence.parquet")
    pd.DataFrame(doc_rows, columns=["doc_idx", "kind", "token_offset", "n_tokens"]).to_parquet(out / "documents.parquet")
    ptab = pd.DataFrame([{k: v for k, v in p.items() if k not in ("cross",)} for p in people])
    ptab["partners"] = ptab["partners"].apply(lambda x: list(map(int, x)))
    ptab["n_cross"] = [len(p["cross"]) for p in people]
    ptab.to_parquet(out / "people.parquet")
    pd.DataFrame({"city": meta["cities"], "load_full_design": [meta["city_load"][c] for c in meta["cities"]],
                  "load_this_run": [run_load[c] for c in meta["cities"]]}).to_parquet(out / "cities.parquet")
    shutil.copy(HERE / "templates.json", out / "templates.json")
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # measured budget for this run, and the full design calibrated with measured tokens per item
    n_exp = sum(v for (pid, a, is_qa), v in exp_counter.items() if not is_qa)
    n_fill = sum(p["n_filler"] for p in people)
    n_qa = sum(p["n_qa"] for p in people)
    tpe = (kind_tokens["fact"] + kind_tokens["cross"]) / max(1, n_exp)
    tpf = kind_tokens["filler"] / max(1, n_fill)
    tpq = kind_tokens["qa"] / max(1, n_qa)
    rows = []
    for p in full:
        c = planned_counts(p, cfg)
        rows.append((p, c, c["facts"] * tpe + c["filler"] * tpf + c["qa"] * tpq))
    calibrated = _budget_table(rows)
    calibrated["tokens_by_kind"] = {"facts": round(calibrated["facts"] * tpe),
                                    "filler": round(calibrated["filler"] * tpf),
                                    "qa": round(calibrated["qa"] * tpq)}
    measured = {"n_people": len(people), "people_per_cell_split": k, "n_docs": doc_idx, "tokens_total": offset,
                "exposures": n_exp, "balancing_name_mentions": n_fill, "qa_pairs": n_qa,
                "tokens_by_kind": dict(kind_tokens), "docs_by_kind": dict(kind_docs),
                "docs_by_context_type": dict(ctx_docs),
                "tokens_per_exposure": round(tpe, 2), "tokens_per_filler_doc": round(tpf, 2),
                "tokens_per_qa": round(tpq, 2), "shards": shard_id + 1}
    name_tok = [len(tok.encode(" " + p["name"]).ids) for p in people[:500]]
    city_tok = [len(tok.encode(" " + c).ids) for c in meta["cities"]]
    budget = {"generator_version": cfg["generator_version"], "templates_version": tpl["version"],
              "seed": cfg["seed"], "fraction": fraction, "estimate_full_before_writing": est_full,
              "measured_this_run": measured, "full_design_calibrated": calibrated,
              "tokens_per_name_mean": round(float(np.mean(name_tok)), 2),
              "tokens_per_city_mean": round(float(np.mean(city_tok)), 2),
              "city_load_full_design": spread(meta["city_load"]), "city_load_this_run": spread(run_load),
              "chance_ah_uniform_mean": round(float(np.mean([p["chance_ah"] for p in people])), 4),
              "chance_ah_freq_mean": round(float(np.mean([p["chance_ah_freq"] for p in people])), 4),
              "cities_multi_token": int(sum(n > 1 for n in city_tok)),  # evaluation needs 0 (answer = 1 token)
              "eos_id": eos_id, "vocab_size": tok.get_vocab_size(),
              "seconds": round(time.time() - t0, 1)}
    with open(out / "budget.json", "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)
    if not quiet:
        print(f"\n=== MEASURED, this run ({len(people):,} people) ===")
        print(f"docs {doc_idx:,} | tokens {offset / 1e6:,.2f} M | exposures {n_exp:,} | balancing {n_fill:,} | QA {n_qa:,}")
        print(f"tokens per exposure {tpe:.1f} | per balancing doc {tpf:.1f} | per QA {tpq:.1f}")
        print(f"tokens per full name {budget['tokens_per_name_mean']} | per city {budget['tokens_per_city_mean']}")
        if budget["cities_multi_token"]:
            print(f"WARNING: {budget['cities_multi_token']} cities are not single tokens; evaluation will refuse")
        print("docs by context:", dict(ctx_docs))
        print_city_spread("city mentions, this run", run_load)
        print(f"chance-level AH: uniform {budget['chance_ah_uniform_mean']:.3f} | "
              f"frequency-weighted {budget['chance_ah_freq_mean']:.3f}")
        print_budget("FULL DESIGN, calibrated with measured tokens per item", calibrated)
        print(f"\nwritten to {out} in {budget['seconds']} s")
    return budget


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "config_v1.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of people per cell (0.01 = dry run)")
    ap.add_argument("--budget-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tpl = load_templates()
    if not args.budget_only and not args.out:
        ap.error("--out is required unless --budget-only")
    generate(cfg, tpl, Path(args.out) if args.out else Path("."), args.fraction, args.budget_only, args.overwrite)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
