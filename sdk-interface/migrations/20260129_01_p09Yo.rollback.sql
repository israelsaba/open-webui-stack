-- Rollback initial schema

DROP INDEX IF EXISTS idx_created_at;
DROP INDEX IF EXISTS idx_deleted_at;
DROP INDEX IF EXISTS idx_md5;
DROP TABLE IF EXISTS research_hashes;
