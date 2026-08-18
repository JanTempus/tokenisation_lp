# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT4o regex
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L101-L111
# https://github.com/openai/tiktoken/blob/4560a889/tiktoken_ext/openai_public.py#L101-L114

# This regex could be made more efficient. If I was the one working on this encoding, I would
# have done a few other things differently too, e.g. I think you can allocate tokens more
# efficiently across languages.
GPT4O_SPLIT_PATTERN = "|".join([
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
])

# Craig's version of a pattern
FULL_PATTERN         = r"'(?i:[sdmt]|ll|ve|re)| ?[\p{Lu}]+(?=[\p{Lu}][\p{Ll}])| ?[\p{Lu}]?[\p{Ll}]+| ?[\p{Lu}]+| ?[\p{Lt}\p{Lm}\p{Lo}]+|\p{N}{1,3}(?=(?:\p{N}{3})*(?:\P{N}|$))| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+"

# the version from the v1 ArXiV paper
# it turns out the lack of an optional space in front of punctuation really messed up the Renyi
# score, as it resulting in a bunch of spaces by themsevles
# so added it for the v2 version of the paper.
ULTIMATE_PATTERN_PARTS_V1 = [
    r" ?(?:\p{L}\p{M}*)+['’](?:\p{L}\p{M}*)+",                  # contraction, allow curly apostrophe
    r"_(?:\p{Ll}\p{M}*)+",                                      # snake_case, with underscore at front TODO: support __init__?
    r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # optional space, uppercase followed by upper and then lower case letter, i.e. the XML in XMLHttpRequest
    r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",                    # optional space, optional uppercase, one or more lowercase i.e. Http or http
    r" ?(?:\p{Lu}\p{M}*)+",                                     # all uppercase acronym CONSTANT
    r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",                       # titlecase, modifier, or other (those without case) letters
    r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))", # numbers
    r"(?:[\p{P}\p{S}]\p{M}*)+",                    # optional space, punctuation and symbols
    r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+",               # whitespace, Is this what there originally going for?
    r"(?:[\p{Z}\p{C}]\p{M}*)+",                    # separator or control with combining marks, note that \s includes \p{Z} plus \r and \n from \p{C}, so put \p{C} after \s ones
    r"\p{M}+",                                     # marks shoud be attatched to some \P{M}, just for incorrect utf-8
    r".+"                                          # left over, should be empty or there is a regex bug
]
ULTIMATE_PATTERN_V1 = "|".join(ULTIMATE_PATTERN_PARTS_V1)


# and the version for the v2 paper, avoiding the extra single spaces
# by adding an optional space to punctuation and symbols
ULTIMATE_PATTERN_PARTS = [
    r" ?(?:\p{L}\p{M}*)+['’](?:\p{L}\p{M}*)+",                  # contraction, allow curly apostrophe
    r"_(?:\p{Ll}\p{M}*)+",                                      # snake_case, with underscore at front TODO: support __init__?
    r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # optional space, uppercase followed by upper and then lower case letter, i.e. the XML in XMLHttpRequest
    r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",                    # optional space, optional uppercase, one or more lowercase i.e. Http or http
    r" ?(?:\p{Lu}\p{M}*)+",                                     # all uppercase acronym CONSTANT
    r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",                       # titlecase, modifier, or other (those without case) letters
    r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))", # numbers
    r" ?(?:[\p{P}\p{S}]\p{M}*)+",                  # optional space, punctuation and symbols
    r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+",               # whitespace, Is this what there originally going for?
    r"(?:[\p{Z}\p{C}]\p{M}*)+",                    # separator or control with combining marks, note that \s includes \p{Z} plus \r and \n from \p{C}, so put \p{C} after \s ones
    r"\p{M}+",                                     # marks shoud be attatched to some \P{M}, just for incorrect utf-8
    r".+"                                          # left over, should be empty or there is a regex bug
]
ULTIMATE_PATTERN = "|".join(ULTIMATE_PATTERN_PARTS)

# is it all whitespace bytes
ALL_WHITESPACE_BYTES = rb"^\s+$"


# which bits should be allowed to be merged in a supermerge
 # TODO: is this unused?
SUPERWORD_PARTS = [
b" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",             # contraction, allow curly apostrophe
b"_(?:\p{Ll}\p{M}*)+",                                      # snake_case, with underscore at front TODO: support __init__?
b" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",                    # optional space, optional uppercase, one or more lowercase i.e. Http or http
b" ?(?:\p{Lu}\p{M}*)+",                                     # all uppercase acronym CONSTANT
b" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",                       # titlecase, modifier, or other (those without case) letters
]

SUPERWORD_PATTERN = b"^(" + b"|".join(SUPERWORD_PARTS) + b")$"


# what I used for the COLM submission
# note it is a bytes regex
# and note that the final * is a bug!!!, no it isn't
ORIGINAL_MERGE_PATTERN = rb"^[ _'a-zA-Z]*[a-zA-Z][ _'a-zA-Z]*$"

# the reviwer didn't like the a-zA-Z
# to do that, we'll need to cast back to a string before use
# - match the entire string
# - lookahead ensures there is at least one letter
# - otherwise can have spaces, underscores, apostrophes, or curly apostrophes
# for reference, this is exactly what was used in the ablation training
IMPROVED_MERGE_PATTERN = r"^(?=.+\p{L})(?:\p{L}\p{M}*|[ _'\u2019])+$"

# suggestion from reviewer, which is the same if you don't have leading \p{M}
# IMPROVED_MERGE_PATTERN = r"^(?=.*\p{L})[\p{L}\p{M} _'\u2019]+$"
