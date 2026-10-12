# Test fixtures

- `exa_search.json`, `tavily_search.json`: real responses from one small query each (2026-10-12). Image and favicon URLs are removed from the Exa results.
- `forexfactory_items.json`: items in the output format of the Apify actor `xtracto/forexfactory-calendar`. The first item is the real sample from the actor's README. The others are written in the same format to cover US releases and edge cases: a holiday, a tentative time, an unreleased event and a repeated id.
- `forexfactory_items_real.json`: real output of the Apify actor `xtracto/forexfactory-calendar` from a one-week probe run on 2026-10-12 (19 items, USD, medium and high impact, 2026-09-10 to 2026-09-17).
