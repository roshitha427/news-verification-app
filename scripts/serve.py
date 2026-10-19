"""Start the local Waitress server with a concise Atlas startup diagnostic."""
import os

from pymongo.errors import ServerSelectionTimeoutError
from waitress import serve

from backend.app import create_server_app


def main():
    try:
        app = create_server_app()
    except ServerSelectionTimeoutError as error:
        message = str(error)
        if 'SSL handshake failed' in message or 'TLSV1_ALERT_INTERNAL_ERROR' in message:
            print('MongoDB Atlas TLS negotiation failed on port 27017. The backend stopped before Flask opened port 5000.')
            print('Check the Atlas cluster, current public IP in Network Access, and outbound TCP 27017. If those are correct, test another trusted network and check for VPN, proxy, antivirus, or carrier TLS filtering.')
            print('Do not disable TLS verification or change the database password for this error.')
        else:
            print('MongoDB Atlas could not select a server. Check the cluster status, current public IP in Network Access, and outbound TCP 27017.')
        print(r'For a focused diagnostic, run: .\.venv\Scripts\python.exe .\scripts\check_mongo.py')
        return 1

    port = int(os.getenv('PORT', '5000'))
    print(f'AI-Powered Fake News Detection and Verification System: http://127.0.0.1:{port}')
    serve(app, host='127.0.0.1', port=port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
