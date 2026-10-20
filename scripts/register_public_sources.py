from backend.app import create_app
from backend.verification import DEFAULT_SOURCES
from urllib.parse import urlparse

if __name__=='__main__':
    app=create_app()
    store=app.extensions['store']
    for name,url,description in DEFAULT_SOURCES:
        domain=urlparse(url).hostname.removeprefix('www.')
        if domain not in ('factly.in','gov.in','nic.in'):
            continue
        if not store.one('sources',{'domain':domain}):
            store.save('sources',dict(name=name,url=url,domain=domain,description=description,enabled=True))
    print('Dataset and public sources registered; existing enabled/disabled settings preserved.')
    app.extensions['analysis_executor'].shutdown(wait=True)
