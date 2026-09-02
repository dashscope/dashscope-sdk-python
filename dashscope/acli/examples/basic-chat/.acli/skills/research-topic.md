---
name: research-topic
description: Research a topic on the web and produce a brief with source URLs
arguments: [topic]
---

Use the `web_search` tool to research "{topic}":

1. Start with one broad search for `{topic}` (max_results=5)
2. Based on the results, refine with 1-2 more specific searches if needed
   (e.g. `{topic} latest progress`, `{topic} benchmarks`, `{topic} comparison`)
3. If you find conflicting information, search for supporting evidence before drawing conclusions

Output format:
- A 50-100 word overview at the top
- 3-5 bullet points, each ending with [source](url)
- If opinions differ, add a separate "Controversies" section

Base every point on the search results; do not invent details that do not appear in the sources.
