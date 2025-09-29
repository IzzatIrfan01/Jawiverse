Back-end for jawiverse

1. Classical Jawi → Rumi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"classical2rumi\",\"text\":\"ڤڠاجارن\"}"

2. Rumi → Classical Jawi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"rumi2classical\",\"text\":\"pengajaran\"}"

3. Modern Jawi → Rumi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"modern2rumi\",\"text\":\"ڤمبچأن\"}"

4. Rumi → Modern Jawi
curl -X POST "http://127.0.0.1:8000/transliterate" -H "Content-Type: application/json" -d "{\"model\":\"rumi2modern\",\"text\":\"pembacaan\"}"
