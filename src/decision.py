def choose_direction(prob_long, prob_short, min_prob_long, min_prob_short, auc_long, auc_short, min_auc):
    """Returns (direction, auc_score, conflict).

    A side signals when its probability reaches its threshold AND its model passes the AUC gate. The long and short labels
    are mutually exclusive (forward return above +x versus below -x), so when both sides signal at once at least one model
    is wrong on that bar: no direction is chosen and `conflict` is True.
    """
    long_ok = prob_long >= min_prob_long and auc_long >= min_auc
    short_ok = prob_short >= min_prob_short and auc_short >= min_auc
    if long_ok and short_ok:
        return None, 0.5, True
    if long_ok:
        return "long", auc_long, False
    if short_ok:
        return "short", auc_short, False
    return None, 0.5, False
