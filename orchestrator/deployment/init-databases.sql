SELECT 'CREATE DATABASE wander'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'wander')\gexec
