"""
Global configuration for Feature Engineering.
""""""
Global configuration for Feature Engineering.
"""

# ==========================================================
# Merge Keys
# ==========================================================

GROUP_COLS = [
    "id_student",
    "code_module",
    "code_presentation",
]

# ==========================================================
# Assessment
# ==========================================================

PASS_MARK = 40

# ==========================================================
# Time Windows (days)
# ==========================================================

EARLY_PERIOD = 30
MID_PERIOD = 90

# ==========================================================
# Numerical Stability
# ==========================================================

EPSILON = 1e-6

# ==========================================================
# Activity Weights
# ==========================================================

ACTIVITY_WEIGHTS = {

    # Navigation
    "homepage": 0.20,
    "dualpane": 0.50,

    # Reading Content
    "page": 0.60,
    "subpage": 0.60,
    "sharedsubpage": 0.60,

    # Learning Materials
    "resource": 1.00,
    "oucontent": 1.00,
    "folder": 1.00,
    "url": 1.00,
    "htmlactivity": 1.00,

    # Knowledge Building
    "glossary": 1.10,
    "dataplus": 1.20,

    # Collaborative Learning
    "forumng": 1.30,
    "ouwiki": 1.30,
    "oucollaborate": 1.30,

    # Assessment
    "questionnaire": 1.50,
    "quiz": 2.00,
    "externalquiz": 2.00,

    # Live Sessions
    "ouelluminate": 1.80,

    # Miscellaneous
    "repeatactivity": 0.80,
}

# ==========================================================
# Merge Keys
# ==========================================================

GROUP_COLS = [
    "id_student",
    "code_module",
    "code_presentation",
]

# ==========================================================
# Assessment
# ==========================================================

PASS_MARK = 40

# ==========================================================
# Time Windows (days)
# ==========================================================

EARLY_PERIOD = 30
MID_PERIOD = 90

# ==========================================================
# Numerical Stability
# ==========================================================

EPSILON = 1e-6

# ==========================================================
# Activity Weights
# ==========================================================

ACTIVITY_WEIGHTS = {

    # Navigation
    "homepage": 0.20,
    "dualpane": 0.50,

    # Reading Content
    "page": 0.60,
    "subpage": 0.60,
    "sharedsubpage": 0.60,

    # Learning Materials
    "resource": 1.00,
    "oucontent": 1.00,
    "folder": 1.00,
    "url": 1.00,
    "htmlactivity": 1.00,

    # Knowledge Building
    "glossary": 1.10,
    "dataplus": 1.20,

    # Collaborative Learning
    "forumng": 1.30,
    "ouwiki": 1.30,
    "oucollaborate": 1.30,

    # Assessment
    "questionnaire": 1.50,
    "quiz": 2.00,
    "externalquiz": 2.00,

    # Live Sessions
    "ouelluminate": 1.80,

    # Miscellaneous
    "repeatactivity": 0.80,
}