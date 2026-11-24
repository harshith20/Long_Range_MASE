"""Invented names for M7 (plan §1.1): people, cities, universities, employers.

All names are built from syllables with a seeded RNG, so a pretrained model (bridge B1) has no prior
knowledge of them. The word pools for people, cities and organisations are disjoint.
"""
from __future__ import annotations

import random

ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "j", "k", "l", "m", "n", "p", "r", "s",
          "st", "t", "tr", "v", "w", "z", "th", "sh", "kl", "pr"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "m", "nd", "rk", "x"]
CITY_SUFFIXES = ["holm", "burg", "ford", "wick", "haven", "stad", "mere", "port", "dale", "gate", "ton", "field"]
UNI_PATTERNS = ["{w} University", "University of {w}", "{w} Institute of Technology", "{w} College",
                "{w} State University"]
EMPLOYER_SUFFIXES = ["Industries", "Labs", "Systems", "Group", "Logistics", "Foods", "Energy", "Media",
                     "Works", "Analytics", "Partners", "Robotics"]
MAJORS = ["Physics", "Chemistry", "Biology", "Mathematics", "Computer Science", "Economics", "History",
          "Philosophy", "Linguistics", "Psychology", "Sociology", "Geology", "Astronomy", "Architecture",
          "Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Medicine", "Law",
          "Music", "Fine Arts", "Literature", "Political Science", "Anthropology", "Statistics",
          "Nursing", "Agriculture", "Marine Biology", "Journalism", "Accounting", "Finance", "Education",
          "Geography", "Neuroscience", "Archaeology", "Theology", "Pharmacy", "Ecology", "Dentistry",
          "Veterinary Science"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _word(rng: random.Random, n_syll: int) -> str:
    parts = []
    for i in range(n_syll):
        coda = rng.choice(CODAS) if i == n_syll - 1 else ""
        parts.append(rng.choice(ONSETS) + rng.choice(VOWELS) + coda)
    return "".join(parts).capitalize()


def _unique_words(rng: random.Random, n: int, syll: tuple[int, int], taken: set[str]) -> list[str]:
    out: list[str] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > n * 1000:
            raise RuntimeError("could not generate enough unique words")
        w = _word(rng, rng.randint(*syll))
        if len(w) < 4 or w in taken:
            continue
        taken.add(w)
        out.append(w)
    return out


def make_lexicon(seed: int, n_cities: int, n_universities: int, n_employers: int,
                 n_first: int, n_middle: int, n_last: int) -> dict:
    """Return disjoint invented word lists. Deterministic for a given seed and sizes."""
    rng = random.Random(seed)
    taken: set[str] = set()
    first = _unique_words(rng, n_first, (2, 2), taken)
    middle = _unique_words(rng, n_middle, (2, 3), taken)
    last = _unique_words(rng, n_last, (2, 3), taken)
    city_roots = _unique_words(rng, n_cities, (1, 2), taken)
    cities = []
    for root in city_roots:
        c = root + rng.choice(CITY_SUFFIXES)
        while c in taken:  # extremely rare; change suffix
            c = root + rng.choice(CITY_SUFFIXES)
        taken.add(c)
        cities.append(c)
    uni_words = _unique_words(rng, n_universities, (2, 3), taken)
    universities = [rng.choice(UNI_PATTERNS).format(w=w) for w in uni_words]
    emp_words = _unique_words(rng, n_employers, (2, 3), taken)
    employers = [f"{w} {rng.choice(EMPLOYER_SUFFIXES)}" for w in emp_words]
    # every organisation sits in a fixed city
    uni_city = {u: rng.choice(cities) for u in universities}
    emp_city = {e: rng.choice(cities) for e in employers}
    return {"first": first, "middle": middle, "last": last, "cities": cities,
            "universities": universities, "university_city": uni_city,
            "employers": employers, "employer_city": emp_city, "majors": list(MAJORS)}


def random_date(rng: random.Random) -> str:
    year = rng.randint(1940, 2002)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{day} {MONTHS[month - 1]} {year}"
