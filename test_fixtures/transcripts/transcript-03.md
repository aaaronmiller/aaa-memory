Human: Database choice?

Assistant: PostgreSQL for relational data, TimescaleDB for time-series. Decision: Dual-database architecture.

Human: Migration plan?

Assistant: Decision: Online migration using pg_dump/pg_restore with read replicas.