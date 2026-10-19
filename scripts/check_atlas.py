"""Test Atlas connectivity without printing its URI or credentials."""
from pathlib import Path
import sys
from uuid import uuid4
from dotenv import dotenv_values
from pymongo import MongoClient


def main():
    config=dotenv_values(Path(__file__).resolve().parents[1]/'backend'/'.env')
    uri=config.get('MONGODB_URI')
    if not uri:
        print('Atlas is not configured. Set MONGODB_URI and DATABASE_BACKEND=mongo in backend/.env.')
        return 2
    try:
        with MongoClient(uri,serverSelectionTimeoutMS=10000,tls=True) as client:
            client.admin.command('ping')
            collection=client[config.get('MONGODB_DATABASE','veritas')]['connection_checks']
            marker=uuid4().hex
            try:
                collection.insert_one({'_id':marker,'purpose':'Veritas connection test'})
                if collection.find_one({'_id':marker}) is None:
                    raise RuntimeError('Read-after-write failed')
            finally:
                collection.delete_one({'_id':marker})
        print('Atlas connection successful. Authentication, ping, write, read, and deletion passed.')
        return 0
    except Exception as exc:
        print('Atlas check failed ('+type(exc).__name__+'). Check database-user permissions, URL-encoded credentials, cluster status, and Atlas IP access list. Credentials were not printed.')
        return 1


if __name__=='__main__':
    sys.exit(main())
