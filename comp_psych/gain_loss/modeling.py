"""
created 26.1.9

utilities for modeling gain/loss data

@author: cgrossman
"""

def get_param_names(model_name):
    """Return the ordered Stan parameter names for a gain_loss model variant.

    Parameters
    ----------
    model_name : str
        One of 'q', 'q_a_win_lose', 'q_a_win_lose_loss_gain',
        'q_a_win_lose_loss_gain_forget' (matching a .stan file under
        `gain_loss.config.MODEL_DIR`).

    Returns
    -------
    list of str
        Parameter names, in the order used to index fitted model output.

    Raises
    ------
    ValueError
        If `model_name` is not a recognized model variant.
    """
    if model_name == 'q':
        return ['a', 'beta']
    elif model_name == 'q_a_win_lose':
        return ['a_win', 'a_lose', 'beta']
    elif model_name == 'q_a_win_lose_loss_gain':
        return ['a_win_gain', 'a_lose_gain', 'a_win_loss', 'a_lose_loss', 'beta']
    elif model_name == 'q_a_win_lose_loss_gain_forget':
        return ['a_win_gain', 'a_lose_gain', 'a_win_loss', 'a_lose_loss', 'forget', 'beta']
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
