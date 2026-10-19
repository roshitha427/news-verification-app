"""Check the configured MongoDB connection without printing credentials."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError, ServerSelectionTimeoutError


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / 'backend' / '.env')
    backend = os.getenv('DATABASE_BACKEND', 'sqlite').strip().lower()
    uri = os.getenv('MONGODB_URI', '').strip()
    database = os.getenv('MONGODB_DATABASE', '').strip()

    if backend != 'mongo':
        print('MongoDB is not selected. Set DATABASE_BACKEND=mongo in this terminal or backend/.env.')
        return 1
    if not uri.startswith(('mongodb://', 'mongodb+srv://')):
        print('MONGODB_URI is missing or is not a MongoDB URI. The URI value was not displayed.')
        return 1
    if not database:
        print('MONGODB_DATABASE is empty. Set it to the Atlas database name.')
        return 1

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=8000, tls=True)
        client.admin.command('ping')
        print(f'MongoDB connection succeeded. Backend: mongo. Database: {database}.')
        return 0
    except ConfigurationError:
        print('MongoDB could not parse the URI or resolve its Atlas SRV record. Check the copied Atlas URI and DNS.')
    except ServerSelectionTimeoutError as error:
        message = str(error)
        if 'SSL handshake failed' in message or 'TLSV1_ALERT_INTERNAL_ERROR' in message:
            print('Atlas shard hosts resolved, but the TLS handshake failed on port 27017. The request did not reach MongoDB authentication.')
            print('Keep TLS verification enabled. Check that the cluster is active, your current public IP is in Atlas Network Access, and outbound TCP 27017 is allowed.')
            print('If those are correct, retry on another trusted network and check whether a VPN, proxy, antivirus, or carrier network is interfering with TLS traffic.')
        else:
            print('Atlas could not select a server. Check that the cluster is active, your current public IP is allowed in Network Access, and outbound TCP 27017 is permitted.')
    except OperationFailure:
        print('Atlas rejected authentication. Check the Database Access username/password and percent-encode special characters in the URI password.')
    except PyMongoError as error:
        print(f'MongoDB check failed ({type(error).__name__}). The connection string and credentials were not displayed.')
    finally:
        if client is not None:
            client.close()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
