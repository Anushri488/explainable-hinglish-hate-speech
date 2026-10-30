"""Text cleaning for code-mixed Hindi-English (Hinglish) social media posts."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")


def clean_text(text, lowercase=True):
    """Clean one post.

    Steps:
    1. Replace URLs with <url> and @mentions with <user> (they carry no meaning here).
    2. Keep the word inside a hashtag but drop the '#'  (#StopThis -> StopThis).
    3. Shorten a character repeated 3+ times to 2  (bahuuuut -> bahuut).
       Hinglish spelling varies a lot, so this reduces noise.
    4. Lowercase (affects Latin script only; Devanagari has no case).
    5. Collapse extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" <url> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    if lowercase:
        text = text.lower()
    return SPACE_RE.sub(" ", text).strip()


def tokenize_words(text):
    """Whitespace tokenisation. Explanations are reported at this word level."""
    return text.split()
