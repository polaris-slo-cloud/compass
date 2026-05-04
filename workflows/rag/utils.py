import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _compute_f1_single(prediction, truth):
    """Compute F1 between prediction and a single ground truth."""
    pred_tokens = set(normalize_answer(prediction).split())
    truth_tokens = set(normalize_answer(truth).split())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = pred_tokens.intersection(truth_tokens)
    if len(common) == 0:
        return 0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(truth_tokens)
    return 2 * (p * r) / (p + r)


def compute_f1(prediction, ground_truths):
    """
    Compute max F1 score across all ground truth answers.

    Args:
        prediction: Model's predicted answer
        ground_truths: Single answer (str) or list of valid answers

    Returns:
        Maximum F1 score across all ground truths
    """
    if isinstance(ground_truths, str):
        return _compute_f1_single(prediction, ground_truths)
    return max(_compute_f1_single(prediction, gt) for gt in ground_truths)


def compute_em(prediction, ground_truths):
    """Compute exact match (1 if any ground truth matches exactly)."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    normalized_pred = normalize_answer(prediction)
    return int(any(normalized_pred == normalize_answer(gt) for gt in ground_truths))