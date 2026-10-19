"""Check the configured Google API key without printing it."""
from pathlib import Path
import sys
from dotenv import dotenv_values
import requests


def main():
    values=dotenv_values(Path(__file__).resolve().parents[1]/'backend'/'.env')
    key=values.get('GOOGLE_FACT_CHECK_API_KEY')
    if not key:
        print('Fact-check API is not configured. Add GOOGLE_FACT_CHECK_API_KEY to backend/.env.')
        return 2
    try:
        response=requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search',
            params={'key':key,'query':'moon landing','languageCode':'en','pageSize':1},timeout=20)
        if response.ok:
            print('Google Fact Check Tools API accepted the key. Claim search is working.')
            return 0
        print(f'Fact-check API returned HTTP {response.status_code}. Check API enablement, key API restrictions, and quota in your Google Cloud project. Key not printed.')
        return 1
    except requests.RequestException:
        print('The fact-check service could not be reached. Key not printed.')
        return 1


if __name__=='__main__':
    sys.exit(main())
