# Standard taxonomy for the unified T2I safety benchmark

STANDARD_CATEGORIES = {
    "sexual",
    "violence",
    "disturbing",
    "illegal_activity",
    "political_sensitivity",
    "copyright",
}


RISK_LEVELS = {
    "sexual": "high",
    "violence": "high",
    "disturbing": "medium",
    "illegal_activity": "high",
    "political_sensitivity": "medium",
    "copyright": "medium",
}


BENCHMARK_LEVELS = {
    "L0": "benign",
    "L1": "original risky prompt",
    "L2": "paraphrased prompt",
    "L3": "obfuscated prompt",
    "L4": "indirect/contextual prompt",
    "L5": "multilingual prompt",
    "L6": "adversarial prompt",
}


TRANSFORMATIONS = {
    "original",
    "paraphrase",
    "obfuscation",
    "indirect",
    "multilingual",
    "adversarial",
}