from utils.converter import jawi_to_rumi, rumi_to_jawi

def classical_jawi_to_rumi(text: str) -> str:
    return jawi_to_rumi(text)

def rumi_to_classical_jawi(text: str) -> str:
    return rumi_to_jawi(text)
