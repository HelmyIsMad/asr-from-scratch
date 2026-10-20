"""Word/character error rates, implemented with edit distance."""


def edit_distance(ref, hyp):
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def wer(reference, hypothesis):
    return edit_distance(reference.split(), hypothesis.split())


def cer(reference, hypothesis):
    return edit_distance(list(reference), list(hypothesis))


def corpus_wer(refs, hyps):
    errors = sum(wer(r, h) for r, h in zip(refs, hyps))
    words = sum(len(r.split()) for r in refs)
    return errors / max(1, words), errors, words


def corpus_cer(refs, hyps):
    errors = sum(cer(r, h) for r, h in zip(refs, hyps))
    chars = sum(len(r.replace(" ", "")) for r in refs)
    return errors / max(1, chars), errors, chars
