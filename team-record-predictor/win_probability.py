"""
Win Probability Predictor based on Box Plus-Minus (BPM)
=======================================================
Calibrated using 2025-26 NCAA Men's Basketball data.

Model equation:
    team_rating = average BPM of players on the roster
    P(Team A wins) = 1 / (1 + exp(-k * (rating_A - rating_B)))

Two modes:
  - Simple:   plain average of BPM scores, k = 0.4511
  - Weighted: minutes-weighted average,     k = 0.4675

Both modes cancel the league-average baseline because it appears
symmetrically in both team ratings when taking the difference.

RMSE ~0.12 win-pct, correlation ~0.73 vs actual 2026 records (284 teams).
"""

import math


# Calibrated from 2025-26 MBB season (284 teams matched)
K_SIMPLE   = 0.4511   # for plain average BPM
K_WEIGHTED = 0.4675   # for minutes-weighted BPM


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def team_rating(bpm_scores: list[float],
                minutes: list[float] | None = None) -> float:
    """
    Compute a single team rating from player BPM scores.

    Parameters
    ----------
    bpm_scores : list of float
        Box Plus-Minus score for each player you want to include.
    minutes : list of float or None
        Minutes played by each player (same order as bpm_scores).
        If provided, the rating is minutes-weighted; otherwise plain average.

    Returns
    -------
    float
        Team rating (weighted or plain average BPM).
    """
    if not bpm_scores:
        raise ValueError("bpm_scores must be a non-empty list.")

    if minutes is not None:
        if len(minutes) != len(bpm_scores):
            raise ValueError("bpm_scores and minutes must have the same length.")
        total_min = sum(minutes)
        if total_min <= 0:
            raise ValueError("Total minutes must be > 0.")
        return sum(b * m for b, m in zip(bpm_scores, minutes)) / total_min
    else:
        return sum(bpm_scores) / len(bpm_scores)


def win_probability(team_a_bpm: list[float],
                    team_b_bpm: list[float],
                    team_a_minutes: list[float] | None = None,
                    team_b_minutes: list[float] | None = None) -> dict:
    """
    Predict the probability that Team A beats Team B.

    Parameters
    ----------
    team_a_bpm : list of float
        BPM scores for each player on Team A you want to include.
    team_b_bpm : list of float
        BPM scores for each player on Team B you want to include.
    team_a_minutes : list of float or None
        Minutes for Team A players. If given (for both teams), uses
        minutes-weighted ratings and the weighted k coefficient.
        If None, uses plain average BPM.
    team_b_minutes : list of float or None
        Minutes for Team B players (same length as team_b_bpm).

    Returns
    -------
    dict with keys:
        'team_a_rating'   – computed rating for Team A
        'team_b_rating'   – computed rating for Team B
        'bpm_diff'        – team_a_rating - team_b_rating
        'team_a_win_prob' – P(Team A wins), 0-1
        'team_b_win_prob' – P(Team B wins), 0-1
        'mode'            – 'weighted' or 'simple'

    Examples
    --------
    # Simple mode — just pass BPM lists
    >>> result = win_probability([8.5, 7.2, 6.1, 5.8, 4.9],
    ...                          [7.0, 6.5, 5.5, 4.8, 3.9])
    >>> print(result['team_a_win_prob'])

    # Weighted mode — also pass minutes
    >>> result = win_probability(
    ...     team_a_bpm=[9.0, 7.5, 6.2, 5.0, 4.3],
    ...     team_b_bpm=[8.0, 6.8, 5.9, 4.7, 3.5],
    ...     team_a_minutes=[850, 780, 700, 650, 600],
    ...     team_b_minutes=[900, 820, 710, 630, 580],
    ... )
    """
    use_weighted = (team_a_minutes is not None) and (team_b_minutes is not None)

    rating_a = team_rating(team_a_bpm, team_a_minutes if use_weighted else None)
    rating_b = team_rating(team_b_bpm, team_b_minutes if use_weighted else None)

    k = K_WEIGHTED if use_weighted else K_SIMPLE
    diff = rating_a - rating_b
    p_a = _sigmoid(k * diff)

    return {
        "team_a_rating":   round(rating_a, 4),
        "team_b_rating":   round(rating_b, 4),
        "bpm_diff":        round(diff, 4),
        "team_a_win_prob": round(p_a, 4),
        "team_b_win_prob": round(1.0 - p_a, 4),
        "mode":            "weighted" if use_weighted else "simple",
    }


def print_matchup(team_a_bpm: list[float],
                  team_b_bpm: list[float],
                  team_a_minutes: list[float] | None = None,
                  team_b_minutes: list[float] | None = None,
                  team_a_name: str = "Team A",
                  team_b_name: str = "Team B") -> None:
    """
    Pretty-print a matchup prediction.

    Parameters
    ----------
    team_a_bpm, team_b_bpm : list of float
        BPM scores for each team's players.
    team_a_minutes, team_b_minutes : list of float or None
        Optional minutes for weighted mode.
    team_a_name, team_b_name : str
        Display names.
    """
    result = win_probability(team_a_bpm, team_b_bpm, team_a_minutes, team_b_minutes)

    print("=" * 50)
    print(f"Matchup: {team_a_name} vs {team_b_name}")
    print(f"Mode: {result['mode']} BPM")
    print("-" * 50)
    print(f"{team_a_name:25s}  rating: {result['team_a_rating']:+.2f}")
    print(f"{team_b_name:25s}  rating: {result['team_b_rating']:+.2f}")
    print(f"BPM differential (A - B): {result['bpm_diff']:+.2f}")
    print("-" * 50)
    print(f"{team_a_name} win probability: {result['team_a_win_prob'] * 100:.1f}%")
    print(f"{team_b_name} win probability: {result['team_b_win_prob'] * 100:.1f}%")
    print("=" * 50)


# ------------------------------------------------------------------ #
#  Quick-reference: what BPM differential means in win probability
# ------------------------------------------------------------------ #
def diff_to_prob_table() -> None:
    """Print a lookup table of BPM differential → win probability."""
    print("\nBPM Diff → Win Probability (Team A, simple-average mode)")
    print("-" * 40)
    print(f"{'Diff':>8}  {'P(A wins)':>10}  {'P(B wins)':>10}")
    for diff in [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6]:
        p = _sigmoid(K_SIMPLE * diff)
        print(f"{diff:>+8.1f}  {p*100:>9.1f}%  {(1-p)*100:>9.1f}%")
    print()


# ------------------------------------------------------------------ #
#  Demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Example 1 – simple mode (just BPM scores)
    print_matchup(
        team_a_bpm=[9.2, 8.1, 7.5, 6.3, 5.8, 4.9],
        team_b_bpm=[7.8, 6.9, 6.2, 5.7, 5.1, 4.0],
        team_a_name="Duke-like Roster",
        team_b_name="Mid-Major Roster",
    )

    print()

    # Example 2 – weighted mode (BPM + minutes)
    print_matchup(
        team_a_bpm=[9.2, 8.1, 7.5, 6.3, 5.8],
        team_b_bpm=[7.8, 6.9, 6.2, 5.7, 5.1],
        team_a_minutes=[900, 850, 800, 700, 650],
        team_b_minutes=[850, 800, 780, 720, 680],
        team_a_name="Team A (weighted)",
        team_b_name="Team B (weighted)",
    )

    print()
    diff_to_prob_table()
