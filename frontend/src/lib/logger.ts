/**
 * Structured JSON logger for the playground UI.
 *
 * Every entry is one JSON object on the console: timestamp, level, message and
 * the caller's context (ids, key state). `debug` is emitted only in dev builds.
 */

export const LogLevel = {
  DEBUG: 'debug',
  INFO: 'info',
  WARN: 'warn',
  ERROR: 'error',
} as const
export type LogLevel = (typeof LogLevel)[keyof typeof LogLevel]

export type LogContext = Record<string, unknown>

function serializeError(value: unknown): unknown {
  if (value instanceof Error) {
    return { name: value.name, message: value.message }
  }
  return value
}

function emit(level: LogLevel, message: string, context: LogContext = {}): void {
  const entry: LogContext = { ts: new Date().toISOString(), level, message }
  for (const [key, value] of Object.entries(context)) {
    entry[key] = serializeError(value)
  }
  console[level](JSON.stringify(entry))
}

export const logger = {
  debug(message: string, context?: LogContext): void {
    if (import.meta.env.DEV) emit(LogLevel.DEBUG, message, context)
  },
  info(message: string, context?: LogContext): void {
    emit(LogLevel.INFO, message, context)
  },
  warn(message: string, context?: LogContext): void {
    emit(LogLevel.WARN, message, context)
  },
  error(message: string, context?: LogContext): void {
    emit(LogLevel.ERROR, message, context)
  },
}
