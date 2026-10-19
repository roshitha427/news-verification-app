from backend.app import create_app

if __name__=='__main__':
    app=create_app()
    store=app.extensions['store']
    if not store.one('sources',{'domain':'factly.in'}):
        store.save('sources',dict(name='Factly',url='https://factly.in',domain='factly.in',description='India · Fact checking',enabled=True))
    print('Public fact-check source registered; existing source access settings preserved.')
    app.extensions['analysis_executor'].shutdown(wait=True)
