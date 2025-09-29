from utils.mapping import JAWI_TO_RUMI, RUMI_TO_JAWI

def jawi_to_rumi(text: str) -> str:
    """Convert Jawi text to Rumi using simple mapping"""
    result = ""
    for char in text:
        if char in JAWI_TO_RUMI:
            result += JAWI_TO_RUMI[char]
        else:
            result += char  # keep unknown chars
    return result

def rumi_to_jawi(text: str) -> str:
    """Convert Rumi text to Jawi (basic rule-based)"""
    result = ""
    i = 0
    while i < len(text):
        # Handle digraphs first (e.g., 'ng', 'ny', 'sy', 'gh')
        if text[i:i+2] in RUMI_TO_JAWI:
            result += RUMI_TO_JAWI[text[i:i+2]]
            i += 2
        elif text[i] in RUMI_TO_JAWI:
            result += RUMI_TO_JAWI[text[i]]
            i += 1
        else:
            result += text[i]
            i += 1
    return result
