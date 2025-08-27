import joblib
import pandas as pd
import re  # NEW: for a tiny numeric hint check

MODEL_PATH = "models/nba_game_outcome_model.joblib"

# Exact features used in training
FEATURES = [
    "FG_PCT_home", "FG_PCT_away",
    "FT_PCT_home", "FT_PCT_away",
    "AST_home", "AST_away",
    "REB_home", "REB_away"
]

# Typical pregame (season-average) ranges to catch out-of-distribution inputs
RANGES = {
    "FG_PCT_home": (0.30, 0.65),
    "FG_PCT_away": (0.30, 0.65),
    "FT_PCT_home": (0.60, 0.95),
    "FT_PCT_away": (0.60, 0.95),
    "AST_home": (15.0, 35.0),
    "AST_away": (15.0, 35.0),
    "REB_home": (35.0, 60.0),
    "REB_away": (35.0, 60.0),
}

# --- Behavior toggles ---
STRICT = True  # True = refuse out-of-range; False = only warn

# Some runners send an empty line to stdin at startup.
# This flag lets us ignore the very first empty input we see.
FIRST_EMPTY_SEEN = False

# NEW: heuristic to detect if the line even looks numeric (contains a digit or a dot)
NUMERIC_HINT = re.compile(r"[0-9.]")

def ask_float(prompt, min_val=None, max_val=None):
    """Ask for a float with optional min/max validation (basic, not 'typical range')."""
    global FIRST_EMPTY_SEEN
    while True:
        try:
            s = input(prompt)
        except EOFError:
            print("\nNo input received. If this is a VS Code Task, run from the Terminal instead.")
            continue

        s = s.strip()

        # Suppress one phantom empty line injected by some runners
        if s == "" and not FIRST_EMPTY_SEEN:
            FIRST_EMPTY_SEEN = True
            continue

        # NEW: Ignore runner noise that isn't even attempting a number (no digits/dot)
        if not NUMERIC_HINT.search(s):
            continue

        if s == "":
            print("No input detected. Please type a number (e.g., 0.45 for 45%).")
            continue

        try:
            v = float(s)
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 0.45 for 45%).")
            continue

        if min_val is not None and v < min_val:
            print(f"Please enter a value >= {min_val}. Try again.")
            continue
        if max_val is not None and v > max_val:
            print(f"Please enter a value <= {max_val}. Try again.")
            continue
        return v

def warn_if_out_of_range(name, val):
    lo, hi = RANGES[name]
    if val < lo or val > hi:
        print(
            f"Warning: {name} = {val} is outside the typical pregame range [{lo}, {hi}]. "
            "Results may be less reliable."
        )

def enforce_range(name, val):
    """If STRICT, reprompt until val is within typical range; else just warn."""
    lo, hi = RANGES[name]
    if lo <= val <= hi:
        return val
    if STRICT:
        print(f"{name} = {val} is outside [{lo}, {hi}]. Please re-enter.")
        while True:
            new_val = ask_float(f"{name} (must be between {lo} and {hi}): ")
            if lo <= new_val <= hi:
                return new_val
            print(f"Still outside [{lo}, {hi}]. Try again.")
    else:
        warn_if_out_of_range(name, val)
        return val

def collect_inputs():
    """Prompt user for all 8 inputs and return a one-row DataFrame."""
    print("\nEnter projected stats for a single matchup.")
    print("Percentages must be decimals between 0 and 1 (e.g., 45% = 0.45).")

    FG_PCT_home = ask_float("FG_PCT_home (0-1): ", 0, 1)
    FG_PCT_away = ask_float("FG_PCT_away (0-1): ", 0, 1)
    FT_PCT_home = ask_float("FT_PCT_home (0-1): ", 0, 1)
    FT_PCT_away = ask_float("FT_PCT_away (0-1): ", 0, 1)

    # Assists/rebounds as season averages (floats)
    AST_home    = ask_float("AST_home (assists per game): ", 0)
    AST_away    = ask_float("AST_away (assists per game): ", 0)
    REB_home    = ask_float("REB_home (rebounds per game): ", 0)
    REB_away    = ask_float("REB_away (rebounds per game): ", 0)

    # Enforce/warn typical ranges
    FG_PCT_home = enforce_range("FG_PCT_home", FG_PCT_home)
    FG_PCT_away = enforce_range("FG_PCT_away", FG_PCT_away)
    FT_PCT_home = enforce_range("FT_PCT_home", FT_PCT_home)
    FT_PCT_away = enforce_range("FT_PCT_away", FT_PCT_away)
    AST_home    = enforce_range("AST_home", AST_home)
    AST_away    = enforce_range("AST_away", AST_away)
    REB_home    = enforce_range("REB_home", REB_home)
    REB_away    = enforce_range("REB_away", REB_away)

    row = [{
        "FG_PCT_home": FG_PCT_home,
        "FG_PCT_away": FG_PCT_away,
        "FT_PCT_home": FT_PCT_home,
        "FT_PCT_away": FT_PCT_away,
        "AST_home": AST_home,
        "AST_away": AST_away,
        "REB_home": REB_home,
        "REB_away": REB_away,
    }]
    return pd.DataFrame(row, columns=FEATURES)

def main():
    # Load the model
    print(f"Loading model from {MODEL_PATH}...\n")
    model = joblib.load(MODEL_PATH)

    # Loop so you can try multiple predictions without restarting
    while True:
        df = collect_inputs()

        # Predict
        pred_class = int(model.predict(df)[0])
        proba_home = float(model.predict_proba(df)[0][1])
        proba_away = 1.0 - proba_home

        label = "Home win" if pred_class == 1 else "Away win"
        print(f"\nPredicted: {label} | P(home win) = {proba_home:.3f} | P(away win) = {proba_away:.3f}\n")

        again = input("Try another prediction? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
