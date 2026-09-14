#!/usr/bin/env python3
"""
Generate fully synthetic veterinary consultation data for the two-stage
Sporadic Cystitis (SC) pipeline.

This produces the three raw CSVs the real pipeline expects
(`dev-utd-predicted.csv`, `dev-all-syndromes.csv`, `test-all-syndromes.csv`),
with the same column schema, encoding, and approximate class-prevalence
pattern described in the paper -- but every row is fabricated from scratch
by combining template sentence fragments. No real clinical note, patient
record, or identifier appears anywhere in this script or its output.

This exists so the rest of the codebase (data preparation, Stage-1 gate,
Stage-2 CV, single-stage baselines, two-stage evaluation) can be run and
inspected end-to-end without access to the private, real annotation data.
Numbers produced from this synthetic data are meaningless and will not
resemble the paper's reported results -- they only demonstrate that the
pipeline runs.

Usage:
    python tools/generate_synthetic_data.py
    python tools/generate_synthetic_data.py --output_dir data/ann04062026 --seed 0

To run the pipeline on your own (real or otherwise) data instead, just
replace the three CSVs this script writes with files sharing the same
column schema -- everything downstream reads from there unchanged.
"""

import argparse
import os
import random

import pandas as pd

COLUMNS = [
    'consult_table_id', 'patient_species', 'patient_sex', 'text',
    'utd_spor_cyst', 'utd_other', 'utd_journey', 'urine_collected',
    'cytology', 'enteropathy_acute_diar', 'enteropathy_other',
    'skin_sup_pyo', 'skin_other', 'other',
]

OTHER_UTD_SUBTYPES = ['FIC', 'FLUTD', 'urolithiasis', 'recurrent cystitis', 'urethral obstruction']

SPECIES = ['Dog', 'Cat']
SEXES = ['Female', 'Male', 'Unknown']

# ---------------------------------------------------------------------------
# Template fragments -- entirely author-written for this generator, chosen
# only to be structurally similar to a veterinary consult note (signalment,
# complaint, exam, assessment, plan). None of this is derived from, or
# resembles, any specific real record.
# ---------------------------------------------------------------------------

SC_COMPLAINTS = [
    "presented for a {age}-year-old {species_lower} with sudden-onset straining to urinate",
    "owner reports frequent small-volume urination over the past {days} days",
    "brought in for pollakiuria and occasional blood-tinged urine, first noticed {days} days ago",
    "history of vocalising in the litter tray with no other systemic signs",
]
SC_EXAM = [
    "Bladder small and mildly painful on palpation, otherwise unremarkable exam",
    "No fever, no lethargy, bright and alert on exam; mild lower abdominal discomfort",
    "Urinalysis shows mild haematuria, no crystals seen, no other systemic findings",
]
SC_ASSESSMENT = [
    "Assessment: sporadic cystitis, likely idiopathic given no other UTD features on exam",
    "Diagnosis: sporadic cystitis; single isolated episode, no history of recurrence",
    "Impression: sporadic (first-episode) cystitis, no evidence of obstruction or calculi",
]
SC_PLAN = [
    "Plan: course of amoxicillin-clavulanate, recheck in 5-7 days if signs persist.",
    "Plan: analgesia and increased water intake encouraged, antibiotics prescribed empirically.",
    "Plan: short antibiotic course, owner advised to return if straining recurs.",
]

OTHER_UTD_COMPLAINTS = {
    'FIC': [
        "presented for recurrent stress-associated lower urinary tract signs",
        "third episode of pollakiuria this year, owner notes recent house move",
    ],
    'FLUTD': [
        "generalised lower urinary tract signs with intermittent inappropriate urination",
        "waxing-and-waning urinary signs over several months",
    ],
    'urolithiasis': [
        "presented for stranguria with palpable firm structures on bladder palpation",
        "radiographs requested to rule out urinary calculi given recurrent signs",
    ],
    'recurrent cystitis': [
        "fourth presentation this year for cystitis-type signs, previously resolved with antibiotics",
        "owner reports recurring urinary signs every few months despite treatment",
    ],
    'urethral obstruction': [
        "presented collapsed with a distended, firm bladder and inability to pass urine",
        "straining unproductively for over 12 hours, suspected blocked cat",
    ],
}
OTHER_UTD_ASSESSMENT = {
    'FIC': "Assessment: feline idiopathic cystitis (FIC), recurrent stress-associated presentation.",
    'FLUTD': "Assessment: feline lower urinary tract disease (FLUTD), multifactorial.",
    'urolithiasis': "Assessment: urolithiasis confirmed on imaging, calculi visible in bladder.",
    'recurrent cystitis': "Assessment: recurrent cystitis, this is a repeat presentation, not a first episode.",
    'urethral obstruction': "Assessment: urethral obstruction, emergency decompression required.",
}
OTHER_UTD_PLAN = {
    'FIC': "Plan: environmental enrichment advised, analgesia, stress reduction; antibiotics not indicated.",
    'FLUTD': "Plan: dietary management trial, recheck in 2 weeks.",
    'urolithiasis': "Plan: surgical or dietary dissolution discussed depending on stone type.",
    'recurrent cystitis': "Plan: extended antibiotic course and referral for underlying-cause work-up.",
    'urethral obstruction': "Plan: urinary catheterisation under sedation, hospitalisation for monitoring.",
}

NON_UTD_TOPICS = [
    ("dental disease", "presented for a routine dental check, mild tartar build-up noted",
     "Assessment: grade 2 periodontal disease.", "Plan: scale and polish scheduled."),
    ("vaccination", "brought in for annual vaccination and general health check",
     "Assessment: healthy adult, no abnormalities detected.", "Plan: vaccination administered, next visit in 12 months."),
    ("skin allergy", "presented for pruritus and mild erythema over the dorsum",
     "Assessment: suspected atopic dermatitis.", "Plan: trial of antihistamines and medicated shampoo."),
    ("orthopaedic", "presented for intermittent hindlimb lameness after exercise",
     "Assessment: suspected mild cruciate strain.", "Plan: rest advised, recheck in 2 weeks if not improved."),
    ("gastroenteritis", "presented with a single episode of vomiting yesterday, otherwise well",
     "Assessment: mild self-limiting gastroenteritis.", "Plan: bland diet for 48 hours, monitor at home."),
    ("otitis", "presented with head shaking and malodorous discharge from the left ear",
     "Assessment: otitis externa, likely yeast overgrowth.", "Plan: ear cleaning and topical medication dispensed."),
]


def _rand_signalment(rng):
    age = rng.randint(1, 14)
    days = rng.choice([1, 2, 3, 4, 5, 7, 10])
    species = rng.choice(SPECIES)
    return age, days, species


def make_sc_text(rng):
    age, days, species = _rand_signalment(rng)
    complaint = rng.choice(SC_COMPLAINTS).format(age=age, days=days, species_lower=species.lower())
    exam = rng.choice(SC_EXAM)
    assessment = rng.choice(SC_ASSESSMENT)
    plan = rng.choice(SC_PLAN)
    return f"{species}, {age}yo. {complaint.capitalize()}. {exam}. {assessment} {plan}", species


def make_other_utd_text(rng, subtype):
    age, days, species = _rand_signalment(rng)
    complaint = rng.choice(OTHER_UTD_COMPLAINTS[subtype]).format(age=age, days=days)
    assessment = OTHER_UTD_ASSESSMENT[subtype]
    plan = OTHER_UTD_PLAN[subtype]
    return f"{species}, {age}yo. {complaint.capitalize()}. {assessment} {plan}", species


def make_non_utd_text(rng):
    age, days, species = _rand_signalment(rng)
    topic, complaint, assessment, plan = rng.choice(NON_UTD_TOPICS)
    return f"{species}, {age}yo. {complaint.capitalize()}. {assessment} {plan}", species


def make_row(rng, consult_id, label):
    """label in {'sc', 'other_utd', 'non_utd'}"""
    if label == 'sc':
        text, species = make_sc_text(rng)
        utd_spor_cyst, utd_other = 1.0, None
    elif label == 'other_utd':
        subtype = rng.choice(OTHER_UTD_SUBTYPES)
        text, species = make_other_utd_text(rng, subtype)
        utd_spor_cyst, utd_other = 0.0, subtype
    else:
        text, species = make_non_utd_text(rng)
        utd_spor_cyst, utd_other = 0.0, None

    utd_journey = 1.0 if rng.random() < 0.03 else None
    other_misc = rng.choice([None, None, None, 'incidental finding', 'senior wellness check'])

    return {
        'consult_table_id': consult_id,
        'patient_species': species,
        'patient_sex': rng.choice(SEXES),
        'text': text,
        'utd_spor_cyst': utd_spor_cyst,
        'utd_other': utd_other,
        'utd_journey': utd_journey,
        'urine_collected': 1.0 if (label != 'non_utd' and rng.random() < 0.6) else None,
        'cytology': float(rng.randint(1, 5)) if (label != 'non_utd' and rng.random() < 0.3) else None,
        'enteropathy_acute_diar': None,
        'enteropathy_other': None,
        'skin_sup_pyo': None,
        'skin_other': None,
        'other': other_misc,
    }


def make_split(rng, n, sc_frac, other_utd_frac, min_sc=1, min_other_utd=1, id_start=1):
    """Build one split's rows with the requested approximate class prevalence."""
    n_sc = max(min_sc, round(n * sc_frac))
    n_other_utd = max(min_other_utd, round(n * other_utd_frac))
    n_non_utd = max(0, n - n_sc - n_other_utd)

    labels = ['sc'] * n_sc + ['other_utd'] * n_other_utd + ['non_utd'] * n_non_utd
    rng.shuffle(labels)

    rows = [make_row(rng, id_start + i, lbl) for i, lbl in enumerate(labels)]
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ann04062026')
    parser.add_argument('--output_dir', type=str, default=default_out,
                         help='Where to write the 3 synthetic CSVs (default: data/ann04062026)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_train', type=int, default=200, help='rows in dev-utd-predicted.csv (train)')
    parser.add_argument('--n_dev', type=int, default=100, help='rows in dev-all-syndromes.csv (dev)')
    parser.add_argument('--n_test', type=int, default=100, help='rows in test-all-syndromes.csv (test)')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Train split: enriched for UTD relevance (mirrors the real train file's
    # VetBERT-guided oversampling), roughly matching the paper's ~48%/19%/33% split.
    train_df = make_split(rng, args.n_train, sc_frac=0.48, other_utd_frac=0.19,
                           min_sc=5, min_other_utd=5, id_start=90_000_001)

    # Dev/test: natural prevalence (rare SC, rarer Other-UTD), but with a
    # floor so downstream stratified CV / gate-threshold code never sees an
    # empty minority class.
    dev_df = make_split(rng, args.n_dev, sc_frac=0.04, other_utd_frac=0.02,
                         min_sc=5, min_other_utd=3, id_start=91_000_001)
    test_df = make_split(rng, args.n_test, sc_frac=0.04, other_utd_frac=0.02,
                          min_sc=5, min_other_utd=3, id_start=92_000_001)

    out_paths = {
        'dev-utd-predicted.csv': train_df,
        'dev-all-syndromes.csv': dev_df,
        'test-all-syndromes.csv': test_df,
    }
    for fname, df in out_paths.items():
        path = os.path.join(args.output_dir, fname)
        # Matches the real files' encoding (see prepare_ann04062026*.py).
        df.to_csv(path, index=False, encoding='latin-1')
        sc_n = int((df['utd_spor_cyst'] == 1.0).sum())
        other_n = int(df['utd_other'].notna().sum())
        print(f"Wrote {path}  ({len(df)} rows: {sc_n} SC, {other_n} Other-UTD, "
              f"{len(df) - sc_n - other_n} non-UTD)")

    print("\nThis is entirely synthetic data -- see data/ann04062026/README.md")
    print("Next: python scripts/data_preparation/prepare_ann04062026_SC_data.py")


if __name__ == '__main__':
    main()
