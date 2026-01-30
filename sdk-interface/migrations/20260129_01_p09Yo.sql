-- 
-- depends: 

CREATE TABLE research_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    md5 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    created_by TEXT,
    updated_at TEXT,
    updated_by TEXT,
    deleted_at TEXT,
    deleted_by TEXT
);

CREATE INDEX idx_md5 ON research_hashes(md5);
CREATE INDEX idx_deleted_at ON research_hashes(deleted_at);
CREATE INDEX idx_created_at ON research_hashes(created_at);
