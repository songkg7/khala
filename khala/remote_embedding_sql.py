"""PostgreSQL statements for chunked remote embeddings.

This module is intentionally dependency-free: it owns schema DDL and query
composition, while :mod:`khala.remote_embeddings` owns validation and cursor
orchestration.
"""

from __future__ import annotations

from khala.roles import RoleFilter


def with_event_role_filter(statement: str, role: RoleFilter | None) -> str:
    """Add a validated event-role predicate to a semantic search statement."""
    if role is None:
        return statement
    active_session_filter = "WHERE (to_jsonb(session)->>'archived_at') IS NULL"
    if statement.count(active_session_filter) != 1:
        raise ValueError("semantic search statement has no unique active-session filter")
    return statement.replace(
        active_session_filter,
        "WHERE event.role = %(role)s\n"
        "  AND (to_jsonb(session)->>'archived_at') IS NULL",
        1,
    )

CHUNK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_embedding_chunks (
    event_id TEXT NOT NULL REFERENCES events(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    chunk_start INTEGER NOT NULL,
    chunk_end INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    dimensions SMALLINT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        event_id, provider, model, content_hash, chunker_version, chunk_index
    ),
    CHECK (content_hash <> ''),
    CHECK (chunker_version <> ''),
    CHECK (chunk_hash <> ''),
    CHECK (chunk_count > 0),
    CHECK (chunk_index >= 0 AND chunk_index < chunk_count),
    CHECK (chunk_start >= 0 AND chunk_end > chunk_start),
    CHECK (dimensions = 1536)
)
"""

CHUNK_LOOKUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS event_embedding_chunks_generation_lookup
ON event_embedding_chunks (
    provider, model, chunker_version, event_id, content_hash
)
"""

CHUNK_TABLE_EXISTS_SQL = """
SELECT to_regclass('event_embedding_chunks') IS NOT NULL AS chunk_table_exists
"""

INSERT_CHUNKS_SQL = """
INSERT INTO event_embedding_chunks (
    event_id, provider, model, content_hash, chunker_version,
    chunk_index, chunk_count, chunk_start, chunk_end, chunk_hash,
    dimensions, embedding
)
VALUES (
    %(event_id)s, %(provider)s, %(model)s, %(content_hash)s, %(chunker_version)s,
    %(chunk_index)s, %(chunk_count)s, %(chunk_start)s, %(chunk_end)s, %(chunk_hash)s,
    %(dimensions)s, %(embedding)s::vector
)
ON CONFLICT (
    event_id, provider, model, content_hash, chunker_version, chunk_index
) DO NOTHING
"""

INSERT_LEGACY_FIRST_CHUNK_SQL = """
INSERT INTO event_embeddings (
    event_id, provider, model, dimensions, content_hash, embedding
)
VALUES (
    %(event_id)s, %(provider)s, %(model)s, %(dimensions)s,
    %(content_hash)s, %(embedding)s::vector
)
ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
"""

ACK_COMPLETE_JOB_SQL = """
DELETE FROM embedding_jobs job
USING events event
WHERE job.event_id = %(event_id)s
  AND job.provider = %(provider)s
  AND job.model = %(model)s
  AND job.content_hash = %(content_hash)s
  AND event.id = job.event_id
  AND event.content = %(content)s
  AND (
    octet_length(event.content) = 0
    OR EXISTS (
      SELECT 1
      FROM event_embedding_chunks chunk
      WHERE chunk.event_id = %(event_id)s
        AND chunk.provider = %(provider)s
        AND chunk.model = %(model)s
        AND chunk.content_hash = %(content_hash)s
        AND chunk.chunker_version = %(chunker_version)s
      GROUP BY chunk.event_id, chunk.provider, chunk.model,
               chunk.content_hash, chunk.chunker_version
      HAVING COUNT(*) = MIN(chunk_count)
         AND MIN(chunk_count) = MAX(chunk_count)
         AND MIN(chunk_index) = 0
         AND MAX(chunk_index) = MAX(chunk_count) - 1
         AND MIN(chunk_start) = 0
         AND MAX(chunk_end) = char_length(event.content)
    )
  )
RETURNING job.event_id
"""

_COMPLETE_GENERATIONS_CTE = """
complete_generations AS (
    SELECT c.event_id, c.provider, c.model, c.content_hash, c.chunker_version,
           MAX(c.created_at) AS generation_created_at
    FROM event_embedding_chunks c
    WHERE c.provider = %(provider)s
      AND c.model = %(model)s
      AND c.chunker_version = %(chunker_version)s
    GROUP BY c.event_id, c.provider, c.model, c.content_hash, c.chunker_version
    HAVING COUNT(*) = MIN(c.chunk_count)
       AND MIN(c.chunk_count) = MAX(c.chunk_count)
       AND MIN(c.chunk_index) = 0
       AND MAX(c.chunk_index) = MAX(c.chunk_count) - 1
),
latest_generations AS (
    SELECT DISTINCT ON (generation.event_id)
           generation.event_id, generation.provider, generation.model,
           generation.content_hash, generation.chunker_version
    FROM complete_generations generation
    ORDER BY generation.event_id, generation.generation_created_at DESC,
             generation.content_hash DESC
)
"""

CHUNK_SEMANTIC_SEARCH_SQL = f"""
WITH {_COMPLETE_GENERATIONS_CTE},
chunk_scores AS (
    SELECT latest.event_id,
           MIN(chunk.embedding <=> %(vec)s::vector) AS distance
    FROM latest_generations latest
    JOIN event_embedding_chunks chunk
      ON chunk.event_id = latest.event_id
     AND chunk.provider = latest.provider
     AND chunk.model = latest.model
     AND chunk.content_hash = latest.content_hash
     AND chunk.chunker_version = latest.chunker_version
    GROUP BY latest.event_id
),
canonical_legacy AS (
    SELECT DISTINCT ON (embedding.event_id)
           embedding.event_id, embedding.embedding
    FROM event_embeddings embedding
    WHERE embedding.provider = %(provider)s
      AND embedding.model = %(model)s
      AND NOT EXISTS (
          SELECT 1 FROM latest_generations latest
          WHERE latest.event_id = embedding.event_id
      )
    ORDER BY embedding.event_id, embedding.created_at DESC,
             embedding.content_hash DESC
),
legacy_scores AS (
    SELECT legacy.event_id,
           legacy.embedding <=> %(vec)s::vector AS distance
    FROM canonical_legacy legacy
),
scores AS (
    SELECT event_id, distance FROM chunk_scores
    UNION ALL
    SELECT event_id, distance FROM legacy_scores
),
scored_events AS (
    SELECT event.id, event.session_id, event.role, event.content, event.created_at,
           session.agent, session.user_name, session.user_email,
           COALESCE(recall.count, 0) AS recall_count,
           score.distance AS semantic_distance,
           0 AS lexical_rank,
           0 AS rank
    FROM scores score
    JOIN events event ON event.id = score.event_id
    JOIN sessions session ON session.id = event.session_id
    LEFT JOIN recalls recall ON recall.event_id = event.id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
      AND event.content NOT LIKE '[tool_calls:%%'
),
deduplicated_events AS (
    SELECT scored_events.*,
           row_number() OVER (
               PARTITION BY role, content
               ORDER BY semantic_distance, created_at DESC, id
           ) AS content_rank
    FROM scored_events
),
ranked_events AS (
    SELECT deduplicated_events.*,
           row_number() OVER (
               ORDER BY semantic_distance, id
           ) AS semantic_rank
    FROM deduplicated_events
    WHERE content_rank = 1
)
SELECT id, session_id, role, content, created_at,
       agent, user_name, user_email, recall_count,
       semantic_distance, lexical_rank, semantic_rank, rank
FROM ranked_events
ORDER BY semantic_distance, id
LIMIT %(limit)s
"""

LEGACY_SEMANTIC_SEARCH_SQL = """\
WITH canonical_embeddings AS (
    SELECT DISTINCT ON (embedding.event_id)
           embedding.event_id, embedding.embedding
    FROM event_embeddings embedding
    WHERE embedding.provider = %(provider)s
      AND embedding.model = %(model)s
    ORDER BY embedding.event_id, embedding.created_at DESC,
             embedding.content_hash DESC
),
scored_events AS (
    SELECT event.id, event.session_id, event.role, event.content, event.created_at,
           session.agent, session.user_name, session.user_email,
           COALESCE(recall.count, 0) AS recall_count,
           canonical.embedding <=> %(vec)s::vector AS semantic_distance,
           0 AS lexical_rank,
           0 AS rank
    FROM canonical_embeddings canonical
    JOIN events event ON event.id = canonical.event_id
    JOIN sessions session ON session.id = event.session_id
    LEFT JOIN recalls recall ON recall.event_id = event.id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
      AND event.content NOT LIKE '[tool_calls:%%'
),
deduplicated_events AS (
    SELECT scored_events.*,
           row_number() OVER (
               PARTITION BY role, content
               ORDER BY semantic_distance, created_at DESC, id
           ) AS content_rank
    FROM scored_events
),
ranked_events AS (
    SELECT deduplicated_events.*,
           row_number() OVER (
               ORDER BY semantic_distance, id
           ) AS semantic_rank
    FROM deduplicated_events
    WHERE content_rank = 1
)
SELECT id, session_id, role, content, created_at,
       agent, user_name, user_email, recall_count,
       semantic_distance, lexical_rank, semantic_rank, rank
FROM ranked_events
ORDER BY semantic_distance, id
LIMIT %(limit)s
"""

_EMBEDDED_EVENT_IDS_CTES = f"""
{_COMPLETE_GENERATIONS_CTE},
canonical_legacy AS (
    SELECT DISTINCT ON (embedding.event_id) embedding.event_id
    FROM event_embeddings embedding
    WHERE embedding.provider = %(provider)s
      AND embedding.model = %(model)s
      AND NOT EXISTS (
          SELECT 1 FROM latest_generations latest
          WHERE latest.event_id = embedding.event_id
      )
    ORDER BY embedding.event_id, embedding.created_at DESC,
             embedding.content_hash DESC
),
embedded_event_ids AS (
    SELECT event_id FROM latest_generations
    UNION
    SELECT event_id FROM canonical_legacy
)
"""

CHUNK_SAMPLE_SQL = f"""
WITH {_EMBEDDED_EVENT_IDS_CTES}
SELECT event.id, event.content, event.created_at
FROM events event
JOIN embedded_event_ids embedded ON embedded.event_id = event.id
JOIN sessions session ON session.id = event.session_id
WHERE event.role IN ('user', 'assistant')
  AND length(event.content) > 30
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND event.content NOT LIKE '[tool_calls:%%'
ORDER BY event.created_at DESC, event.id
LIMIT %(limit)s
"""

LEGACY_SAMPLE_SQL = """
WITH canonical_embeddings AS (
    SELECT DISTINCT ON (embedding.event_id) embedding.event_id
    FROM event_embeddings embedding
    WHERE embedding.provider = %(provider)s
      AND embedding.model = %(model)s
    ORDER BY embedding.event_id, embedding.created_at DESC,
             embedding.content_hash DESC
)
SELECT event.id, event.content, event.created_at
FROM canonical_embeddings canonical
JOIN events event ON event.id = canonical.event_id
JOIN sessions session ON session.id = event.session_id
WHERE event.role IN ('user', 'assistant')
  AND length(event.content) > 30
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND event.content NOT LIKE '[tool_calls:%%'
ORDER BY event.created_at DESC, event.id
LIMIT %(limit)s
"""

CHUNK_STATUS_SQL = f"""
WITH {_EMBEDDED_EVENT_IDS_CTES}
SELECT count(DISTINCT event.id) AS count
FROM embedded_event_ids embedded
JOIN events event ON event.id = embedded.event_id
JOIN sessions session ON session.id = event.session_id
WHERE (to_jsonb(session)->>'archived_at') IS NULL
  AND event.content NOT LIKE '[tool_calls:%%'
"""

LEGACY_STATUS_SQL = """
SELECT count(DISTINCT embedding.event_id) AS count
FROM event_embeddings embedding
JOIN events event ON event.id = embedding.event_id
JOIN sessions session ON session.id = event.session_id
WHERE embedding.provider = %(provider)s
  AND embedding.model = %(model)s
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND event.content NOT LIKE '[tool_calls:%%'
"""

ENQUEUE_MISSING_SQL = """
WITH complete_generations AS (
    SELECT chunk.event_id, chunk.content_hash
    FROM event_embedding_chunks chunk
    WHERE chunk.provider = %(provider)s
      AND chunk.model = %(model)s
      AND chunk.chunker_version = %(chunker_version)s
    GROUP BY chunk.event_id, chunk.provider, chunk.model,
             chunk.content_hash, chunk.chunker_version
    HAVING COUNT(*) = MIN(chunk.chunk_count)
       AND MIN(chunk.chunk_count) = MAX(chunk.chunk_count)
       AND MIN(chunk.chunk_index) = 0
       AND MAX(chunk.chunk_index) = MAX(chunk.chunk_count) - 1
)
INSERT INTO embedding_jobs (event_id, provider, model, content_hash)
SELECT event.id, %(provider)s, %(model)s, md5(event.content)
FROM events event
JOIN sessions session ON session.id = event.session_id
WHERE event.role IN ('user', 'assistant')
  AND octet_length(event.content) > 0
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND event.content NOT LIKE '[tool_calls:%%'
  AND NOT EXISTS (
      SELECT 1 FROM complete_generations complete
      WHERE complete.event_id = event.id
  )
  AND (
      NOT EXISTS (
          SELECT 1 FROM event_embeddings legacy
          WHERE legacy.event_id = event.id
            AND legacy.provider = %(provider)s
            AND legacy.model = %(model)s
      )
      OR octet_length(event.content) > %(legacy_full_content_max_bytes)s
  )
  AND NOT EXISTS (
      SELECT 1 FROM embedding_jobs job
      WHERE job.event_id = event.id
        AND job.provider = %(provider)s
        AND job.model = %(model)s
  )
ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
"""
