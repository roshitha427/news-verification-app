"""MongoDB persistence with an explicitly selected SQLite development adapter."""
import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


class Store:
    def __init__(self, config):
        self.kind = config['DATABASE_BACKEND']
        self.lock = threading.RLock()
        if self.kind == 'mongo':
            from pymongo import MongoClient
            if not config.get('MONGODB_URI'):
                raise RuntimeError('MONGODB_URI is required when DATABASE_BACKEND=mongo')
            self.client = MongoClient(config['MONGODB_URI'], serverSelectionTimeoutMS=7000,
                                      connectTimeoutMS=7000, tls=True)
            self.client.admin.command('ping')
            self.db = self.client[config['MONGODB_DATABASE']]
            for collection in ('users', 'reports', 'sources', 'datasets', 'password_resets'):
                self.db[collection].create_index('id', unique=True)
            self.db.users.create_index('email', unique=True)
            self.db.sources.create_index('domain', unique=True)
            self.db.datasets.create_index([('name', 1), ('version', 1)], unique=True)
            self.db.reports.create_index([('owner_id', 1), ('created_at', -1)])
            self.db.reports.create_index('status')
            self.db.password_resets.create_index('user_id')
            self.db.password_resets.create_index('expires_at', expireAfterSeconds=0)
        elif self.kind == 'sqlite':
            path = Path(config['SQLITE_PATH'])
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path), check_same_thread=False)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS documents (collection TEXT, id TEXT, body TEXT, PRIMARY KEY(collection,id))')
            self.connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_email ON documents(json_extract(body,'$.email')) WHERE collection='users'")
            self.connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_source_domain ON documents(json_extract(body,'$.domain')) WHERE collection='sources'")
            self.connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_dataset_version ON documents(json_extract(body,'$.name'),json_extract(body,'$.version')) WHERE collection='datasets'")
            self.connection.commit()
        else:
            raise RuntimeError('DATABASE_BACKEND must be mongo or sqlite')

    def save(self, collection, doc):
        doc = dict(doc)
        doc.setdefault('id', uuid4().hex)
        doc.setdefault('created_at', now())
        if self.kind == 'mongo':
            self.db[collection].replace_one({'id': doc['id']}, doc, upsert=True)
        else:
            with self.lock, self.connection:
                self.connection.execute('INSERT INTO documents VALUES (?,?,?) ON CONFLICT(collection,id) DO UPDATE SET body=excluded.body',
                                        (collection, doc['id'], json.dumps(doc, default=_json_default)))
        return doc

    def find(self, collection, query=None, limit=500):
        query = query or {}
        if self.kind == 'mongo':
            return list(self.db[collection].find(query, {'_id': 0}).sort('created_at', -1).limit(limit))
        # Local adapter only; Atlas uses indexed queries above.
        with self.lock:
            rows = self.connection.execute('SELECT body FROM documents WHERE collection=?', (collection,)).fetchall()
        docs = [json.loads(row[0]) for row in rows]
        return sorted([d for d in docs if all(d.get(k) == v for k, v in query.items())],
                      key=lambda d: d.get('created_at', ''), reverse=True)[:limit]

    def one(self, collection, query):
        docs = self.find(collection, query, limit=1)
        return docs[0] if docs else None

    def delete(self, collection, doc_id):
        if self.kind == 'mongo':
            self.db[collection].delete_one({'id': doc_id})
        else:
            with self.lock, self.connection:
                self.connection.execute('DELETE FROM documents WHERE collection=? AND id=?', (collection, doc_id))

    def take(self, collection, doc_id):
        """Atomically fetch and delete a single-use record."""
        if self.kind == 'mongo':
            return self.db[collection].find_one_and_delete({'id': doc_id}, projection={'_id': 0})
        with self.lock, self.connection:
            row = self.connection.execute('SELECT body FROM documents WHERE collection=? AND id=?',
                                          (collection, doc_id)).fetchone()
            if row is None:
                return None
            self.connection.execute('DELETE FROM documents WHERE collection=? AND id=?', (collection, doc_id))
            return json.loads(row[0])

    def ping(self):
        if self.kind == 'mongo':
            self.client.admin.command('ping')
        else:
            with self.lock:
                self.connection.execute('SELECT 1').fetchone()

    def count(self, collection):
        if self.kind == 'mongo':
            return self.db[collection].count_documents({})
        with self.lock:
            return self.connection.execute('SELECT count(*) FROM documents WHERE collection=?', (collection,)).fetchone()[0]
