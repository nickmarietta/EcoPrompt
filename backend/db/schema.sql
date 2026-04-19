CREATE TABLE IF NOT EXISTS prompt_runs (
  id SERIAL PRIMARY KEY,
  raw_prompt TEXT NOT NULL,
  task_type TEXT NOT NULL,
  target_model TEXT,
  status TEXT NOT NULL DEFAULT 'completed',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_results (
  id SERIAL PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES prompt_runs(id) ON DELETE CASCADE,
  optimized_prompt TEXT NOT NULL,
  issues_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  changes_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  model_name TEXT,
  latency_ms INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_scores (
  id SERIAL PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES prompt_runs(id) ON DELETE CASCADE,
  clarity_before REAL,
  clarity_after REAL,
  specificity_before REAL,
  specificity_after REAL,
  conciseness_before REAL,
  conciseness_after REAL,
  overall_before REAL,
  overall_after REAL
);

CREATE TABLE IF NOT EXISTS prompt_examples (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  avg_rating REAL,
  agreement_ratio REAL,
  kind TEXT,
  topic TEXT,
  cluster_description TEXT,
  source_dataset TEXT
);

CREATE INDEX IF NOT EXISTS idx_prompt_runs_created_at
  ON prompt_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_runs_task_type
  ON prompt_runs(task_type);

CREATE INDEX IF NOT EXISTS idx_prompt_examples_topic
  ON prompt_examples(topic);

CREATE INDEX IF NOT EXISTS idx_prompt_examples_kind
  ON prompt_examples(kind);