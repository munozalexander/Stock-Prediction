get-nytimes-articles
====================

Python tools for getting data from the New York Times Article API. Retrieves JSON from the API, stores it, parses it into a TSV file.

New York Times Article API Docs: http://developer.nytimes.com/docs/read/article_search_api_v2

Requesting an API Key for the Times API: http://developer.nytimes.com/docs/reference/keys

## Recent Updates
- use config file instead of manually editing lines in main .py file
- check whether file exists before trying to parse it (See Issue #1)
- changed references to CSV to TSV since that's what really gets produced
- make script smart about whether or not to keep fetching for that day (i.e., stop when no more articles)
- solve KeyError issues in parse module
- get better info from API calls with errors

## Dependencies
Python v2.7 (not tested on any others)
Modules:
- urllib2 (HTTPError)
- json
- datetime
- time
- sys
- ConfigParser
- logging

## Why store the JSON files? Why not just parse them?
The New York Times is nice enough to allow programmatic access to its articles, but that doesn't mean I should query the API every time I want data. Instead, I query it once and cache the raw data, lessening the burden on the Times API. Then, I parse that raw data into whatever format I need - in this case a tab-delimited file with only some of the fields - and leave the raw data alone. Next time I have a research question that relies on the same articles, I can just re-parse the stored JSON files into whatever format helps me answer my new question.

## Usage
Set your variables in the config file (copy settings_example.cfg to settings.cfg).

```python getTimesArticles.py```

## Planned improvements
- capture and re-request page after intermittent "504: Bad Gateway" errors
- make script smart about running multi-day processes (i.e., respect the API limit and wait when more than 10K calls are needed)
