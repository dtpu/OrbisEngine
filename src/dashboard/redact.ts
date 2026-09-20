const SECRET_PATTERNS: Array<[RegExp, string]> = [
  [/\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi, 'Bearer [REDACTED]'],
  [/\bAKIA[0-9A-Z]{16}\b/g, '[REDACTED AWS KEY]'],
  [/\b(sk|pk)-[A-Za-z0-9_-]{16,}\b/g, '[REDACTED KEY]'],
  [
    /\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[=:]\s*["']?[^\s,"']+/gi,
    '$1=[REDACTED]',
  ],
  [/([?&](?:token|key|secret|signature|credential)=)[^&#\s]+/gi, '$1[REDACTED]'],
];

export function redactSecrets(value: string): string {
  return SECRET_PATTERNS.reduce(
    (result, [pattern, replacement]) => result.replace(pattern, replacement),
    value,
  );
}
