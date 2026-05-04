# ripper_tw.py
# =============================================================================
# RIPPER-TW: Time-Weighted RIPPER
# =============================================================================
# Internal modification of the RIPPER algorithm for blockchain fraud detection.
#
# WHAT WE CHANGE:
#   Standard RIPPER uses FOIL information gain to decide which condition to add
#   to a rule at each step. The formula counts examples equally regardless of
#   when they occurred. We replace raw counts with WEIGHTED counts, where each
#   transaction's weight is determined by its time step: recent transactions
#   count more, old ones count less.
#
# WHY:
#   Blockchain fraud patterns evolve over time (concept drift). A condition
#   that was highly predictive at time step 1 may no longer be relevant at
#   step 34. By giving more voting power to recent transactions during rule
#   construction, learned rules reflect current fraud behavior rather than
#   historical patterns.
#
# HOW (technical):
#   We replace base_functions.gain_cn (the internal FOIL gain function) with
#   our time-weighted version at runtime using Python module patching.
#   The patched function is active only during .fit(), then restored.
#   This is a genuine internal modification to the algorithm's learning
#   criterion — not a preprocessing step.
#
# WEIGHT FORMULA:
#   w_i = exp(decay * (t_i - t_max))
#   where t_i = time step of example i, t_max = most recent time step
#   - Recent examples (t_i = t_max): weight = exp(0) = 1.0
#   - Older examples: weight approaches 0 as decay increases
#   - decay = 0: all weights = 1.0 → identical to standard RIPPER
#   - decay = 0.1: gentle recency bias
#   - decay = 0.3: moderate recency bias
#   - decay = 1.0: strong recency bias
#
# USAGE:
#   from ripper_tw import RIPPER_TW
#   clf = RIPPER_TW(decay=0.2, k=2, max_rules=30, max_rule_conds=7)
#   clf.fit(X_train, y_train, time_steps=train['time_step'])
#   predictions = clf.predict(X_test)
# =============================================================================

import math
import numpy as np
import pandas as pd
import wittgenstein.base_functions as bf
from wittgenstein import RIPPER

# ─────────────────────────────────────────────
# Store the original gain_cn function so we
# can restore it after training completes.
# ─────────────────────────────────────────────
_original_gain_cn = bf.gain_cn

# ─────────────────────────────────────────────
# Module-level weight storage.
# This dict maps DataFrame index -> float weight.
# Set before fit(), cleared after fit().
# ─────────────────────────────────────────────
_current_weights = None


def _time_weighted_gain_cn(cn, cond_step, rule_covers_pos_idx, rule_covers_neg_idx):
    """
    Time-weighted FOIL information gain — internal replacement for gain_cn.

    Standard gain_cn counts how many positive/negative examples a candidate
    condition covers using len(). We replace len() with weighted sums, so that
    recent transactions contribute more to the gain score than old ones.

    If no weights are active (e.g. during prediction), falls back to the
    original gain_cn automatically.

    Parameters
    ----------
    cn : CatNap
        Internal wittgenstein data structure (unchanged).
    cond_step : Cond
        Candidate condition being evaluated.
    rule_covers_pos_idx : set
        Indices of positive (illicit) examples covered by current rule.
    rule_covers_neg_idx : set
        Indices of negative (licit) examples covered by current rule.

    Returns
    -------
    float
        Weighted FOIL information gain of adding cond_step to the rule.
    """
    global _current_weights

    # No weights set → use original function (safe fallback)
    if _current_weights is None:
        return _original_gain_cn(cn, cond_step, rule_covers_pos_idx, rule_covers_neg_idx)

    weights = _current_weights

    # ── Weighted counts ──────────────────────────────────────────────────
    # Instead of:  p0count = len(rule_covers_pos_idx)
    # We use:      p0count = sum of weights for those indices
    #
    # This means a cluster of 10 old illicit transactions contributes less
    # than a cluster of 5 recent ones, if the recent ones have higher weights.
    # ─────────────────────────────────────────────────────────────────────

    # Weighted count of positives covered by current rule (before adding cond)
    p0count = sum(weights.get(i, 1.0) for i in rule_covers_pos_idx)

    # Weighted count of positives covered by rule + cond (after adding cond)
    pos_after = cn.cond_covers(cond_step, subset=rule_covers_pos_idx)
    p1count = sum(weights.get(i, 1.0) for i in pos_after)

    # Weighted count of negatives covered by current rule (before adding cond)
    n0count = sum(weights.get(i, 1.0) for i in rule_covers_neg_idx)

    # Weighted count of negatives covered by rule + cond (after adding cond)
    neg_after = cn.cond_covers(cond_step, subset=rule_covers_neg_idx)
    n1count = sum(weights.get(i, 1.0) for i in neg_after)

    # Guard: if condition covers no positives, gain is 0
    if p1count <= 0:
        return 0.0

    # ── FOIL gain formula (unchanged, just with weighted counts) ──────────
    # gain = p1 * (log2((p1+1)/(p1+n1+1)) - log2((p0+1)/(p0+n0+1)))
    #
    # Intuition: high gain when p1 is large (captures many illicit) and
    # n1 is small (avoids licit). The difference of logs measures how much
    # better the rule+cond is at distinguishing classes versus rule alone.
    # ─────────────────────────────────────────────────────────────────────
    return p1count * (
        math.log2((p1count + 1) / (p1count + n1count + 1))
        - math.log2((p0count + 1) / (p0count + n0count + 1))
    )


# =============================================================================
# RIPPER_TW CLASS
# =============================================================================

class RIPPER_TW(RIPPER):
    """
    RIPPER with Time-Weighted FOIL Gain (RIPPER-TW).

    Extends standard RIPPER by modifying the internal FOIL information gain
    criterion to account for temporal recency. During rule growing, each
    training example contributes to the gain calculation proportionally to
    its recency weight rather than equally.

    This adaptation is specifically designed for non-stationary data streams
    such as blockchain transaction data, where fraud patterns evolve over time.

    Parameters
    ----------
    decay : float, default=0.1
        Temporal decay rate. Controls how quickly older examples lose influence.
        - decay=0.0 : uniform weights → identical to standard RIPPER
        - decay=0.1 : gentle recency bias (recommended starting point)
        - decay=0.3 : moderate recency bias
        - decay=0.5 : strong recency bias
        Larger values make the model focus almost exclusively on recent steps.
    All other parameters (k, max_rules, max_rule_conds, etc.) are inherited
    from wittgenstein.RIPPER.

    Notes
    -----
    The modification patches wittgenstein.base_functions.gain_cn at runtime
    during .fit() and restores the original function afterward. This ensures
    the change is isolated to training and does not affect other RIPPER instances.
    """

    def __init__(self, decay=0.1, **kwargs):
        """
        Parameters
        ----------
        decay : float
            Temporal decay rate for recency weighting.
        **kwargs
            All standard RIPPER parameters (k, max_rules, max_rule_conds,
            prune_size, dl_allowance, random_state, etc.)
        """
        self.decay = decay
        super().__init__(**kwargs)

    def fit(self, X_train, y_train=None, time_steps=None, **kwargs):
        """
        Fit RIPPER-TW with time-weighted FOIL gain.

        Parameters
        ----------
        X_train : DataFrame
            Feature matrix. Should NOT contain the time_step column.
        y_train : Series or array-like
            Class labels (1=illicit, 0=licit).
        time_steps : Series or array-like, optional
            Time step values for each training example, same length as X_train.
            If None, falls back to standard RIPPER (uniform weights).
        **kwargs
            Additional arguments passed to RIPPER.fit() (e.g. pos_class).

        Returns
        -------
        self
        """
        global _current_weights

        # ── Step 1: Compute time-based weights ───────────────────────────
        weights = self._compute_weights(X_train, time_steps)

        if weights is not None:
            print(f"  [RIPPER-TW] decay={self.decay} | "
                  f"weight range: {min(weights.values()):.4f} – {max(weights.values()):.4f} | "
                  f"mean weight: {np.mean(list(weights.values())):.4f}")
        else:
            print(f"  [RIPPER-TW] No time_steps provided — using uniform weights (standard RIPPER)")

        # ── Step 2: Patch the internal gain function ──────────────────────
        # We replace gain_cn in the base_functions module so that ALL
        # internal calls to it during training use our weighted version.
        _current_weights = weights
        bf.gain_cn = _time_weighted_gain_cn

        try:
            # ── Step 3: Run standard RIPPER training ──────────────────────
            # All internal calls to gain_cn now go through our patched version.
            result = super().fit(X_train, y_train, **kwargs)
        finally:
            # ── Step 4: Always restore original, even if training fails ───
            bf.gain_cn = _original_gain_cn
            _current_weights = None

        return result

    def _compute_weights(self, X_train, time_steps):
        """
        Compute exponential recency weights from time step values.

        Weight formula:
            w_i = exp(decay * (t_i - t_max))

        Examples at the most recent time step get weight 1.0.
        Examples at earlier steps get exponentially smaller weights.

        Parameters
        ----------
        X_train : DataFrame
            Training features (used only for index alignment).
        time_steps : array-like or None
            Time step for each training example.

        Returns
        -------
        dict or None
            Maps DataFrame index -> float weight.
            Returns None if time_steps is None (triggers standard gain).
        """
        if time_steps is None:
            return None

        time_steps = np.array(time_steps)
        t_max = time_steps.max()

        # exp(0) = 1.0 for the most recent step
        # exp(decay * negative_value) < 1.0 for older steps
        raw_weights = np.exp(self.decay * (time_steps - t_max))

        # Map training data index → weight
        weights = dict(zip(X_train.index, raw_weights))
        return weights