# Test fixtures

- `exa_search.json`, `tavily_search.json`: real responses from one small query each (2026-10-12). Image and favicon URLs are removed from the Exa results.
- `forexfactory_items.json`: items in the output format of the Apify actor `xtracto/forexfactory-calendar`. The first item is the real sample from the actor's README. The others are written in the same format to cover US releases, a holiday, a tentative time, an unreleased event and a repeated id. Replace this with a real run's sample once `APIFY_TOKEN` is available.
