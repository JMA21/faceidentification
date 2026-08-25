from __future__ import annotations

import sqlite3
from pathlib import Path

DB_FILE_NAME = "faceid-index.sqlite3"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS photo_roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id INTEGER NOT NULL REFERENCES photo_roots(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    modified_time_ns INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    last_indexed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reference_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reference_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES reference_people(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    modified_time_ns INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    last_indexed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS photo_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    bbox_left REAL NOT NULL,
    bbox_top REAL NOT NULL,
    bbox_right REAL NOT NULL,
    bbox_bottom REAL NOT NULL,
    score REAL NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reference_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_image_id INTEGER NOT NULL REFERENCES reference_images(id) ON DELETE CASCADE,
    bbox_left REAL NOT NULL,
    bbox_top REAL NOT NULL,
    bbox_right REAL NOT NULL,
    bbox_bottom REAL NOT NULL,
    score REAL NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS validation_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES reference_people(id) ON DELETE CASCADE,
    photo_face_id INTEGER NOT NULL REFERENCES photo_faces(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK(state IN ('confirmed', 'rejected', 'manual')),
    source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(person_id, photo_face_id)
);

CREATE TABLE IF NOT EXISTS search_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES reference_people(id) ON DELETE CASCADE,
    threshold REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS match_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES reference_people(id) ON DELETE CASCADE,
    photo_face_id INTEGER NOT NULL REFERENCES photo_faces(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(search_run_id, photo_face_id)
);

CREATE TABLE IF NOT EXISTS engine_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS idx_photos_fingerprint ON photos(fingerprint);
CREATE INDEX IF NOT EXISTS idx_photos_root_path ON photos(root_id, path);
CREATE INDEX IF NOT EXISTS idx_reference_images_status ON reference_images(status);
CREATE INDEX IF NOT EXISTS idx_photo_faces_photo_id ON photo_faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_reference_faces_reference_image_id ON reference_faces(reference_image_id);
CREATE INDEX IF NOT EXISTS idx_validation_person_face ON validation_rules(person_id, photo_face_id);
"""


def resolve_database_path(storage_directory: str | Path) -> Path:
    storage_path = Path(storage_directory)
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path / DB_FILE_NAME


def connect(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> Path:
    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
    return db_path
