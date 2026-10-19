"""Retrieve fact checks; a search match alone never verifies an arbitrary claim."""
from urllib.parse import urlparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
from html import unescape
from concurrent.futures import ThreadPoolExecutor
import re
import threading
import time
import requests
from backend.store import now

DEFAULT_SOURCES = [
    ('Reuters', 'https://www.reuters.com', 'International reporting'),
    ('Associated Press', 'https://apnews.com', 'Independent reporting'),
    ('BBC', 'https://www.bbc.com', 'International reporting'),
    ('AFP Fact Check', 'https://factcheck.afp.com', 'Fact checking'),
    ('Full Fact', 'https://fullfact.org', 'Fact checking'),
    ('Alt News', 'https://www.altnews.in', 'India · Fact checking'),
    ('Factly', 'https://factly.in', 'India · Fact checking'),
    ('PIB Fact Check', 'https://pib.gov.in', 'India · Official information'),
    ('National Portal of India', 'https://www.india.gov.in', 'India · Official information'),
    ('Legislative Department, Government of India', 'https://legislative.gov.in', 'India · Primary legal source'),
    ('India Code', 'https://www.indiacode.nic.in', 'India · Primary legal source'),
]


def trusted_url(url, sources):
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    return parsed.scheme == 'https' and any(
        host == s['domain'] or host.endswith('.' + s['domain']) for s in sources if s.get('enabled', True))


_gdelt_lock = threading.Lock()
_gdelt_cache = {}
_gdelt_last_request = 0.
STOPWORDS = set('this that there their these those which while where when what would could should about after before being because through against between during under over into with from have has had were was are been not will said says also news claim says just more some than then them they your our you its and the for but any all who how can may'.split())
_SEARCH_STOPWORDS = STOPWORDS - {'not', 'no', 'never'}


def gdelt_query(text, sources):
    """Build exact-phrase and entity/value searches without passing query syntax."""
    raw_words = re.findall(r"\d{1,4}|[A-Za-z][A-Za-z0-9'-]{1,}", text[:2500])
    counts = Counter(word.casefold() for word in raw_words
                     if word.casefold() not in _SEARCH_STOPWORDS and len(word) > 1)
    # Keep the user's order so names, negation, dates, and quantities survive.
    ordered = list(dict.fromkeys(word for word in raw_words
                                 if word.casefold() in counts))
    priority = [word for word in ordered if word.isdigit() or
                (word[:1].isupper() and word.casefold() not in _SEARCH_STOPWORDS)]
    terms = list(dict.fromkeys(priority + ordered))[:10]
    domains = [s['domain'] for s in sources if s.get('enabled', True)
               and not s.get('domain', '').endswith('wikipedia.org')
               and re.fullmatch(r'[a-z0-9.-]+', s.get('domain', ''))][:12]
    if len(terms) < 2 or not domains:
        return None
    phrase_terms = ordered[:14]
    compact = ' '.join(terms)
    phrase = '"' + ' '.join(phrase_terms).replace('"', '') + '"' if len(phrase_terms) >= 4 else compact
    domain_clause = '(' + ' OR '.join('domain:' + domain for domain in domains) + ')'
    return f'({phrase} OR ({compact})) {domain_clause}'


def source_reliability(item):
    """A transparent source class; this is not a guarantee of correctness."""
    url = str((item or {}).get('url') or '')
    host = (urlparse(url).hostname or '').lower()
    kind = str((item or {}).get('kind') or '').lower()
    if kind == 'fact_check_review' and (item or {}).get('rating'):
        return 'published_fact_check_with_rating'
    if kind == 'reference_article':
        return 'encyclopedia_background'
    if host.endswith(('.gov', '.gov.in', '.gov.uk', '.gov.au', '.nic.in')):
        return 'official_primary_source'
    if host.endswith('.edu'):
        return 'academic_source'
    if kind == 'fact_check_article':
        return 'fact_check_publisher_page_unrated'
    if host.endswith(('reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'afp.com')):
        return 'established_news_reporting'
    return 'publisher_not_independently_classified'


class _ArticleParser(HTMLParser):
    """Extract page metadata and short paragraph text without storing HTML."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs = []
        self.meta = {}
        self.published_at = None
        self._paragraph = None
        self._title = False
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'noscript', 'svg'):
            self._ignored += 1
        if tag == 'meta':
            key = (attrs.get('property') or attrs.get('name') or '').lower()
            value = attrs.get('content') or ''
            if key in ('description', 'og:description', 'twitter:description') and value:
                self.meta.setdefault(key, value)
            elif key in ('article:published_time', 'datepublished', 'date') and value:
                self.published_at = value
        elif tag == 'time' and not self.published_at:
            self.published_at = attrs.get('datetime')
        elif tag == 'title':
            self._title = True
        elif tag == 'p' and self._paragraph is None:
            self._paragraph = []

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'svg') and self._ignored:
            self._ignored -= 1
        if tag == 'title':
            self._title = False
        elif tag == 'p' and self._paragraph is not None:
            value = re.sub(r'\s+', ' ', ' '.join(self._paragraph)).strip()
            if len(value) >= 50:
                self.paragraphs.append(value)
            self._paragraph = None

    def handle_data(self, data):
        if self._ignored:
            return
        if self._paragraph is not None:
            self._paragraph.append(data)


def _article_excerpt(url, claim, sources):
    """Fetch a bounded excerpt only from an enabled HTTPS source host."""
    if not trusted_url(url, sources):
        return None, None
    try:
        response = requests.get(url, timeout=(3, 5), stream=True, allow_redirects=False,
                                headers={'User-Agent': 'NewsVerificationResearchProject/1.0 (source excerpt reader)'})
        if response.status_code >= 300:
            response.close()
            return None, None
        if 'html' not in response.headers.get('Content-Type', '').lower():
            response.close()
            return None, None
        chunks, total = [], 0
        for chunk in response.iter_content(chunk_size=32768):
            total += len(chunk)
            if total > 512 * 1024:
                break
            chunks.append(chunk)
        encoding = response.encoding or 'utf-8'
        response.close()
        parser = _ArticleParser()
        parser.feed(b''.join(chunks).decode(encoding, errors='replace'))
        from backend.evidence_assessment import claim_tokens
        query_terms = claim_tokens(claim)
        ranked = sorted(parser.paragraphs,
                        key=lambda para: len(query_terms & claim_tokens(para)), reverse=True)
        selected = [para for para in ranked[:4]
                    if len(query_terms & claim_tokens(para)) >= min(2, len(query_terms))]
        excerpt = ' '.join(selected) if selected else (
            parser.meta.get('og:description') or parser.meta.get('description') or
            parser.meta.get('twitter:description') or '')
        excerpt = unescape(re.sub(r'\s+', ' ', excerpt)).strip()
        return excerpt[:1800] or None, parser.published_at
    except (requests.RequestException, ValueError, UnicodeError, TypeError):
        return None, None


def enrich_source_records(verification, claim, sources):
    """Add source classes and, where possible, short text from each original URL."""
    verification = dict(verification or {})
    records = [dict(item) for item in verification.get('evidence') or [] if isinstance(item, dict)]
    for record in records:
        record['reliability'] = source_reliability(record)
    # Public page retrieval is bounded and concurrent so a list of slow or
    # inaccessible publishers cannot make report generation wait serially.
    can_fetch_original = verification.get('provider') != 'Local evidence model'
    targets = ([index for index, record in enumerate(records[:8])
                if record.get('url') and len(str(record.get('snippet') or '')) < 350]
               if can_fetch_original else [])
    if targets:
        with ThreadPoolExecutor(max_workers=4) as executor:
            retrieved = list(executor.map(
                lambda index: _article_excerpt(records[index].get('url', ''), claim, sources), targets))
        for index, (excerpt, published_at) in zip(targets, retrieved):
            record = records[index]
            if excerpt:
                if record.get('snippet'):
                    record['matched_excerpt'] = record['snippet']
                record['snippet'] = excerpt
                record['retrieval_status'] = 'article_excerpt_retrieved'
            else:
                record['retrieval_status'] = 'metadata_only'
            if published_at and not record.get('published_at'):
                record['published_at'] = published_at[:80]
    for record in records:
        if not record.get('retrieval_status'):
            record['retrieval_status'] = 'matched_excerpt_only' if record.get('snippet') else 'metadata_only'
        if record.get('seen_at') and not record.get('published_at'):
            # GDELT's seen date is an observation date, not the article's
            # publication date. Keep it separate so the UI cannot mislabel it.
            record['seen_at'] = record['seen_at']
    enriched = records
    verification['evidence'] = enriched
    return verification


def verify_gdelt(text, sources):
    global _gdelt_last_request
    result = {'status':'unverified', 'checked_at':now(), 'evidence':[], 'provider':'GDELT DOC 2.0',
        'provider_status':'unavailable', 'evidence_type':'related_reporting',
        'explanation':'The public news search could not be reached. The claim remains unverified.',
        'attribution_url':'https://www.gdeltproject.org/'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search for related reporting.')
        return result
    query = gdelt_query(text, sources)
    if query is None:
        result.update(provider_status='query_needed', explanation='For GDELT search, add at least two English keywords about the claim and enable trusted sources. No translation model is installed.')
        return result
    result['search_query'] = query
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    if not _gdelt_lock.acquire(blocking=False):
        result.update(provider_status='busy', explanation='The public news search is busy. Please try again shortly.')
        return result
    try:
        cached = _gdelt_cache.get(cache_key)
        if cached and cached[0] > time.monotonic() - 600:
            return dict(cached[1], cached=True)
        delay = max(0., 5 - (time.monotonic() - _gdelt_last_request))
        if delay:
            time.sleep(delay)
        _gdelt_last_request = time.monotonic()
        # DOC 2.0 searches a rolling three-month window. Context 2.0 is limited
        # to 72 hours and therefore misses many historical claims.
        response = requests.get('https://api.gdeltproject.org/api/v2/doc/doc', params={
            'query':query, 'mode':'ArtList', 'format':'json', 'maxrecords':20,
            'timespan':'3months', 'sort':'HybridRel'}, timeout=(5, 18))
        response.raise_for_status()
        from datetime import datetime, timezone
        seen_urls = set()
        payload = response.json()
        for article in payload.get('articles', []) if isinstance(payload, dict) else []:
            url = article.get('url', '')
            if url in seen_urls or not trusted_url(url, sources):
                continue
            seen_urls.add(url)
            published = None
            seen_date = article.get('seendate') or article.get('seen_at') or ''
            if seen_date:
                try:
                    published = datetime.strptime(seen_date, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    try:
                        published = datetime.fromisoformat(seen_date.replace('Z', '+00:00')).isoformat()
                    except ValueError:
                        pass
            snippet = re.sub(r'\s+', ' ', str(article.get('context') or article.get('sentence') or article.get('snippet') or '')).strip()
            result['evidence'].append({'kind':'related_reporting', 'title':article.get('title', '')[:500],
                'url':url, 'publisher':urlparse(url).hostname, 'claim':'', 'rating':'',
                'reviewed_at':None, 'seen_at':published, 'snippet':snippet[:1200],
                'retrieval_status':'metadata_only',
                'relationship':'GDELT found this article by search terms. Open the original page; a search match is not evidence that it supports or contradicts the claim.'})
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified',
            explanation='Articles from enabled trusted publishers were found in GDELT DOC search. A title or search match alone is not a fact-check verdict.'
                if result['evidence'] else 'No matching reporting from enabled trusted publishers was found in the search window. This does not establish truth or falsehood.')
        if len(_gdelt_cache) > 128:
            _gdelt_cache.clear()
        _gdelt_cache[cache_key] = (time.monotonic(), dict(result))
    except (requests.RequestException, ValueError, TypeError):
        pass
    finally:
        _gdelt_lock.release()
    if result['provider_status'] == 'unavailable':
        # Keep GDELT as the selected provider, but let the project's free
        # fact-check archives and publisher feeds provide a fallback when its
        # endpoint is down or unreachable. A fallback search still never
        # turns a match into an automatic truth verdict.
        try:
            from backend.factcheck_archives import verify_public
            fallback = verify_public(text, sources)
        except Exception:
            fallback = None
        if fallback is not None:
            provider_statuses = [{'name': 'GDELT DOC 2.0', 'status': 'unavailable'}]
            provider_statuses.extend(fallback.get('providers') or [{
                'name': fallback.get('provider', 'Public fact-check search'),
                'status': fallback.get('provider_status', 'unavailable'),
            }])
            if fallback.get('provider_status') == 'searched':
                fallback['provider'] = 'GDELT with public archive/RSS fallback'
                fallback['primary_provider_status'] = 'unavailable'
                fallback['providers'] = provider_statuses
                fallback['explanation'] = 'GDELT could not be reached. ' + fallback.get('explanation', '')
                return fallback
            result['providers'] = provider_statuses
            result['fallback_provider_status'] = fallback.get('provider_status', 'unavailable')
            result['explanation'] = (
                'GDELT and the free fact-check archives/RSS fallback could not be reached. '
                'The claim remains unverified.'
            )
    return result


def _verify_core(text, language, api_key, sources, provider='auto', evidence_path=None):
    if provider == 'local':
        from backend.local_evidence import verify_local
        return verify_local(text, sources, evidence_path)
    if provider == 'public' or (provider == 'auto' and not api_key):
        from backend.factcheck_archives import verify_public
        return verify_public(text, sources)
    if provider == 'rss':
        from backend.public_evidence import verify_rss
        return verify_rss(text, sources)
    if provider == 'gdelt':
        return verify_gdelt(text, sources)
    result = {'status': 'unverified', 'checked_at': now(), 'evidence': [],
              'provider': 'Google Fact Check Tools', 'provider_status': 'not_configured',
              'explanation': 'No live fact-check service is configured. Model predictions are not evidence of truth.'}
    if not text.strip():
        result.update(provider_status='no_claim', explanation='Add a text claim or caption to search for supporting fact checks.')
        return result
    if not api_key or provider == 'none':
        return result
    try:
        response = requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search',
                                params={'query': text[:1000], 'languageCode': language, 'key': api_key, 'pageSize': 10}, timeout=15)
        response.raise_for_status()
        for claim in response.json().get('claims', []):
            for review in claim.get('claimReview', []):
                url = review.get('url', '')
                if trusted_url(url, sources):
                    result['evidence'].append({'claim': claim.get('text', ''), 'title': review.get('title', ''),
                        'url': url, 'publisher': review.get('publisher', {}).get('name', ''),
                        'rating': review.get('textualRating', ''), 'reviewed_at': review.get('reviewDate'),
                        'kind': 'fact_check_review', 'source_type': 'published_fact_check',
                        'relationship': 'Search match — compare the original claim and context'})
        result.update(provider_status='searched', status='needs_review' if result['evidence'] else 'unverified',
                      explanation='Related fact checks found. Review the original claim, date, and context before drawing a conclusion.'
                      if result['evidence'] else 'No matching fact checks from your enabled trusted sources were found. This does not establish that the claim is true or false.')
    except (requests.RequestException, ValueError):
        result.update(provider_status='unavailable', explanation='The fact-check service could not be reached. Verification remains open.')
    return result


def verify_wikipedia(text):
    """Retrieve short encyclopedia excerpts as labeled background references."""
    result = {'name':'Wikipedia reference search', 'status':'unavailable'}
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text[:2500])
    ignored = STOPWORDS | {'current', 'currently', 'today', 'true', 'false', 'fake', 'real', 'verify', 'verified'}
    query = ' '.join(dict.fromkeys(word for word in words if word.lower() not in ignored))[:180]
    if len(query) < 4:
        result['status'] = 'query_needed'
        return result
    try:
        response = requests.get('https://en.wikipedia.org/w/api.php', params={
            'action':'query', 'list':'search', 'srsearch':query, 'srlimit':3,
            'format':'json', 'formatversion':2, 'utf8':1},
            headers={'User-Agent':'NewsVerificationResearchProject/1.0 (evidence retrieval)'}, timeout=(4, 8))
        response.raise_for_status()
        pages = response.json().get('query', {}).get('search', [])
        for page in pages[:3]:
            page_id = page.get('pageid')
            if not isinstance(page_id, int):
                continue
            snippet = re.sub(r'<[^>]+>', '', str(page.get('snippet', '')))
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            result.setdefault('evidence', []).append({
                'kind':'reference_article', 'title':page.get('title', '')[:300],
                'url':f'https://en.wikipedia.org/?curid={page_id}', 'publisher':'Wikipedia',
                'claim':'', 'rating':'', 'reviewed_at':None, 'snippet':snippet[:700],
                'relationship':'Encyclopedia background retrieved by keyword search. It is not a fact-check verdict; confirm current facts with official or independent reporting.'})
        result['status'] = 'searched'
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass
    return result


def verify(text, language, api_key, sources, provider='auto', evidence_path=None):
    result = _verify_core(text, language, api_key, sources, provider, evidence_path)
    # If GDELT DOC 2.0 is reachable but returns no matches, also search the
    # free fact-check archives and publisher RSS feeds so claims are not left
    # with only encyclopedia references.
    if (text.strip() and provider == 'gdelt' and not result.get('evidence')
            and result.get('provider_status') != 'not_configured'
            and not result.get('fallback_provider_status')
            and not result.get('primary_provider_status')):
        try:
            from backend.factcheck_archives import verify_public
            archive_result = verify_public(text, sources)
        except Exception:
            archive_result = None
        if archive_result:
            result.setdefault('providers', []).append({'name':'GDELT DOC 2.0',
                                                       'status':result.get('provider_status', 'unavailable')})
            archive_evidence = archive_result.get('evidence') or []
            result.setdefault('providers', []).extend(archive_result.get('providers') or [{
                'name': archive_result.get('provider', 'Public fact-check search'),
                'status': archive_result.get('provider_status', 'unavailable'),
            }])
            if archive_evidence:
                result['evidence'] = (result.get('evidence') or []) + archive_evidence
                result['evidence'] = result['evidence'][:12]
                result['provider'] = 'GDELT with public archive/RSS fallback'
                result['archive_provider_status'] = archive_result.get('provider_status', 'searched')
                result['status'] = 'needs_review'
                result['explanation'] = (
                    'No matching recent GDELT reporting was found. Related fact-check archive or publisher-feed '
                    'results were found; compare the claim, publication date, and context before drawing a conclusion.'
                )
    # The no-key Wikimedia reference search is a secondary path when the
    # selected evidence source has no passages. GDELT results stay focused on
    # current reporting when it has relevant matches.
    if text.strip() and provider not in ('none', 'local') and (provider != 'gdelt' or not result.get('evidence')):
        reference = verify_wikipedia(text)
        result.setdefault('providers', []).append({'name':reference['name'], 'status':reference['status']})
        if reference.get('evidence'):
            result.setdefault('evidence', []).extend(reference['evidence'])
            result['evidence'] = result['evidence'][:12]
            result['reference_scope'] = 'Encyclopedia excerpts are background references, not current fact-check verdicts.'
    return enrich_source_records(result, text, sources)


def verify_submission(text, language, api_key, sources, provider='auto', evidence_path=None):
    """Search each independent claim separately, then merge the source records."""
    from backend.evidence_assessment import split_candidate_claims

    claims = split_candidate_claims(text)
    if len(claims) <= 1:
        result = verify(text, language, api_key, sources, provider, evidence_path)
        for item in result.get('evidence') or []:
            item.setdefault('candidate_claim_indices', [0] if claims else [])
        result['claim_searches'] = ([{'claim_index': 0, 'claim': claims[0],
            'provider_status': result.get('provider_status', 'unknown'),
            'evidence_count': len(result.get('evidence') or []),
            'search_query': result.get('search_query')}] if claims else [])
        return result

    searches = []
    merged = {}
    provider_rows = {}
    for claim_index, candidate in enumerate(claims):
        found = verify(candidate, language, api_key, sources, provider, evidence_path)
        claim_ids = []
        for row in found.get('evidence') or []:
            if not isinstance(row, dict):
                continue
            url = str(row.get('url') or '').strip()
            key = url.casefold() if url else f"{claim_index}:{row.get('title', '')}:{row.get('snippet', '')}"
            if key not in merged:
                merged[key] = dict(row, candidate_claim_indices=[])
            if claim_index not in merged[key]['candidate_claim_indices']:
                merged[key]['candidate_claim_indices'].append(claim_index)
            claim_ids.append(key)
        searches.append({'claim_index': claim_index, 'claim': candidate,
            'provider_status': found.get('provider_status', 'unknown'),
            'evidence_keys': claim_ids, 'search_query': found.get('search_query')})
        for provider_row in found.get('providers') or [{'name': found.get('provider', 'Unknown'),
                                                         'status': found.get('provider_status', 'unknown')}]:
            name = provider_row.get('name', 'Unknown')
            row = provider_rows.setdefault(name, {'name': name, 'status': provider_row.get('status', 'unknown'),
                                                   'searched_claim_indices': []})
            if provider_row.get('status') == 'searched':
                row['status'] = 'searched'
                row['searched_claim_indices'].append(claim_index)
            elif row['status'] not in ('searched', 'unavailable'):
                row['status'] = provider_row.get('status', 'unknown')

    evidence = list(merged.values())[:12]
    statuses = [item['provider_status'] for item in searches]
    overall_status = ('searched' if 'searched' in statuses else
                      'unavailable' if statuses and all(status == 'unavailable' for status in statuses) else
                      statuses[0] if statuses else 'no_claim')
    # Resolve temporary URL keys into the stable post-deduplication IDs used by
    # the report and by Gemini's citation validation.
    for search in searches:
        search['evidence_ids'] = [index for index, row in enumerate(evidence, start=1)
                                  if search['claim_index'] in row.get('candidate_claim_indices', [])]
        search.pop('evidence_keys', None)
    return {'status': 'needs_review' if evidence else 'unverified',
        'checked_at': now(), 'evidence': evidence,
        'provider': 'Claim-specific evidence search', 'provider_status': overall_status,
        'providers': list(provider_rows.values()), 'claim_searches': searches,
        'search_queries': [item['search_query'] for item in searches if item.get('search_query')],
        'evidence_type': 'claim_specific_reporting',
        'explanation': ('Each factual statement was searched separately. Retrieved reporting is context for review, not a verdict by itself.'
                        if evidence else 'Each factual statement was searched separately, but the configured sources returned no matching excerpts. This does not establish truth or falsehood.'),
        'attribution_url': 'https://www.gdeltproject.org/' if provider == 'gdelt' else None}
