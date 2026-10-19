"""Diagnose Atlas DNS, TLS, and MongoDB connectivity without revealing secrets."""
import os
import socket
import ssl
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from pymongo.uri_parser import parse_uri


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / 'backend' / '.env')
    uri = os.getenv('MONGODB_URI', '').strip()
    if not uri.startswith(('mongodb://', 'mongodb+srv://')):
        print('MONGODB_URI is missing or invalid. Its value was not displayed.')
        return 1

    try:
        parsed = parse_uri(uri)
        nodes = parsed.get('nodelist') or []
    except (PyMongoError, ValueError):
        print('Could not parse the MongoDB URI or resolve its DNS seed records. The URI was not displayed.')
        return 1

    if not nodes:
        print('No MongoDB hosts were found in the configured URI.')
        return 1

    print(f'Found {len(nodes)} MongoDB host(s). Checking verified TLS (certificate checks remain enabled):')
    context = ssl.create_default_context()
    for host, port in nodes:
        try:
            with socket.create_connection((host, port), timeout=6) as raw_socket:
                address = raw_socket.getpeername()[0]
                with context.wrap_socket(raw_socket, server_hostname=host) as tls_socket:
                    print(f'  TLS OK: {host}:{port} -> {address}; {tls_socket.version()}')
        except (OSError, ssl.SSLError, socket.timeout) as error:
            print(f'  TLS FAILED: {host}:{port}; {type(error).__name__}: {error}')

    print('Checking MongoDB ping with the configured URI...')
    client = None
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=8000,
            tls=True,
        )
        client.admin.command('ping')
        print('MongoDB ping succeeded. TLS, server selection, and authentication are working.')
        return 0
    except OperationFailure:
        print('MongoDB rejected authentication. Check the Atlas Database Access username and password.')
    except PyMongoError as error:
        print(f'MongoDB ping failed: {type(error).__name__}. See the TLS results above; no URI or credentials were printed.')
    finally:
        if client is not None:
            client.close()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
