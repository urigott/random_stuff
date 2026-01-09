@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "threshold": [Real],
        "pos_label": [Real, str, "boolean", None],
    },
    prefer_skip_nested_validation=True,
)
def net_benefit_curve(
    y_true,
    y_proba,
    pos_label=None,
):
    """Compute the Net Benefit given a threshold.

    # TODO: Read more in the :ref:`User Guide <net_benefit>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels or :term:`label indicator matrix`. 

    y_score : array-like of shape (n_samples,) 
        Target scores.

    pos_label : int, float, bool or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.


    Returns
    -------
    thresholds : array-like of shape (100,)
        The thresholds used to compute the Net Benefit (0.0 to 0.99)
    net_benefits : array-like of shape (n_thresholds,)
        The Net Benefit for the model at each threshold.
    net_benefits_treat_all : array-like of shape (n_thresholds,)
        The Net Benefit for the treat all strategy at each threshold.

    
    Notes
    -----
    Net benefit is defined as a decision-analytic measure that quantifies the
    clinical value of aprediction model, test, or marker by putting its benefits
    and harms on the same scale. This is done by specifying an exchange rate—
    a clinical judgment about how much a harm (e.g., an unnecessary biopsy) can 
    be tolerated in order to gain a benefit (e.g., detecting a cancer). The net 
    benefit calculation then subtracts weighted harms from benefits to determine 
    whether using the model or test would do more good than harm in practice
    unlike traditional statistics such as sensitivity or AUC that do not directly 
    reflect clinical value.

    Net Benefit = (Sensitivity) - (1 - Specificity) * exchange_rate
    where:
    - Sensitivity = True Positive Rate
    - Specificity = True Negative Rate
    - Exchange Rate = The exchange rate between the benefit and the cost of being False Positive, 
                      which is the odds of the threshold (th / (1-th))
    
    The Net Benefit is then calculated as the difference between the benefit and the cost.

    References
    ----------
    .. [1] `Vickers AJ., Van Calster B., Steyerberg EW. (2016). Net benefit 
            approaches to the evaluation of prediction models, molecular markers, 
            and diagnostic tests. The British Journal of Medicine 352:i6.
            <https://www.bmj.com/content/352/bmj.i6>`_

    .. [2] `Wikipedia entry for the Decision curve analysis
            <https://en.wikipedia.org/wiki/Decision_curve_analysis>`_

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import net_benefit_curve
    >>> import matplotlib.pyplot as plt
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LogisticRegression(solver="newton-cholesky", random_state=0).fit(X, y)
    >>> thresholds, net_benefits, net_benefits_treat_all = net_benefit_curve(y, clf.predict_proba(X)[:, 1])
    >>> plt.plot(thresholds, net_benefits, label='Model')
    >>> plt.plot(thresholds, net_benefits_treat_all, label='Treat all strategy')
    >>> plt.hlines(0, *plt.xlim(), label='Treat none strategy')
    >>> plt.legend()
    >>> plt.show()
    """

    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_proba = check_array(y_proba, ensure_2d=False)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_type != "binary":
        raise NotImplementedError("Net Benefit is only implemented for binary classification.")
    
    prevalence = (y_true == pos_label).mean()

    thresholds = np.linspace(0, 0.99, 100)
    exchange_rates = thresholds / (1 - thresholds)
    positive_cases = (y_true == pos_label)
    
    tps = np.array([np.mean((y_proba > th) & (positive_cases)) for th in thresholds])
    fps = np.array([np.mean((y_proba > th) & (~positive_cases)) for th in thresholds])
    
    net_benefits = tps - fps * exchange_rates
    net_benefits_treat_all = np.array(prevalence - (1 - prevalence) * exchange_rates)
    
    return thresholds, net_benefits, net_benefits_treat_all




