import numpy as np

from fuzzy_logic.core import mf_income, mf_household, mf_workexp, mf_age, mf_dependents, mf_output
from fuzzy_logic.rules import RULES

UNIVERSE = np.linspace(0, 100, 500)
THRESHOLD = 55

SUGENO_CONST = {
    'Tidak_Layak'  : 15.0,
    'Kurang_Layak' : 35.0,
    'Cukup_Layak'  : 62.0,
    'Layak'        : 85.0,
}

def fuzzify(inc, hh, we, age, dep):
    return {
        'Income'     : mf_income(inc),
        'Household'  : mf_household(hh),
        'WorkExp'    : mf_workexp(we),
        'Age'        : mf_age(age),
        'Dependents' : mf_dependents(dep),
    }

def mamdani_inference(fv):
    agg = np.zeros(len(UNIVERSE))
    for rule in RULES:
        inc_l, hh_l, we_l, age_l, dep_l, out_l = rule
        strength = min(
            fv['Income'][inc_l],
            fv['Household'][hh_l],
            fv['WorkExp'][we_l],
            fv['Age'][age_l],
            fv['Dependents'][dep_l],
        )
        if strength == 0:
            continue
        for i, y in enumerate(UNIVERSE):
            agg[i] = max(agg[i], min(strength, mf_output(y)[out_l]))
    return agg

def centroid(agg):
    num = np.sum(UNIVERSE * agg)
    den = np.sum(agg)
    return num / den if den != 0 else 50.0

def fuzzy_mamdani(inc, hh, we, age, dep):
    fv    = fuzzify(inc, hh, we, age, dep)
    agg   = mamdani_inference(fv)
    score = centroid(agg)
    label = 'Layak' if score >= THRESHOLD else 'Tidak Layak'
    return score, label, agg

def sugeno_inference(fv):
    firings = []
    for rule in RULES:
        inc_l, hh_l, we_l, age_l, dep_l, out_l = rule
        strength = min(
            fv['Income'][inc_l],
            fv['Household'][hh_l],
            fv['WorkExp'][we_l],
            fv['Age'][age_l],
            fv['Dependents'][dep_l],
        )
        if strength > 0:
            firings.append((strength, SUGENO_CONST[out_l]))
    return firings

def weighted_average(firings):
    if not firings:
        return 50.0
    num = sum(s * v for s, v in firings)
    den = sum(s for s, _ in firings)
    return num / den if den != 0 else 50.0

def fuzzy_sugeno(inc, hh, we, age, dep):
    fv      = fuzzify(inc, hh, we, age, dep)
    firings = sugeno_inference(fv)
    score   = weighted_average(firings)
    label   = 'Layak' if score >= THRESHOLD else 'Tidak Layak'
    return score, label, firings
