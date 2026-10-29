"""Result lines fit to commit, shared by the runners and collectors here (no dependencies).

    sys.path.insert(0, str(<this folder>)); import records
"""
import re

HOME = re.compile(r'/home/[^/\s"\']+/')
USERHOST = re.compile(r'\b[\w.-]+@(?!127\.0\.0\.1\b)[\w.-]+')
# Console lines worth keeping from a guest run: kernel trouble and anything about revocation or poison.
CONSOLE = re.compile(r'panic|trap|fatal|revok|poison|KDB|Stopped at|exception|signal|lock of', re.I)


def scrub(value):
    """Home paths to ~/, user@host to <user>@<host>, in a string or in every string of a list or dict;
    the rest of each line is kept."""
    if isinstance(value, str):
        return USERHOST.sub('<user>@<host>', HOME.sub('~/', value))
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def leaks(text):
    """Whether TEXT still holds a home path or a user@host string a scrub would have replaced."""
    return '/home/' in text or bool(USERHOST.search(text.replace('<user>@<host>', '')))
