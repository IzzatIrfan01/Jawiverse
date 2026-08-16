# Jawiverse
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/IzzatIrfan01/Jawiverse/blob/main/Llama3_2_(3B)_Jawiverse.ipynb)

curl -X POST "http://localhost:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"rumi2jawi\",\"text\":\"cinta sejati adalah suatu perkara\"}"

curl -X POST "http://localhost:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"jawi2rumi\",\"text\":\"چينتا سجاتي اداله سوات ڤركارا يڠ مرجوع كڤد حقيقة باطن\"}"