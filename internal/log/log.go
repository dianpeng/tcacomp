// Package log is a thin facade over log/slog (TOOL-02): pass and compiler
// code log through this package so nothing outside it imports slog directly
// (CLAUDE.md stack table). The facade keeps compiler records leveled and
// grouped (per-pass, per-node) while leaving the handler swappable.
//
// Logs never touch stdout (RESEARCH Pitfall 6): the default handler writes
// to stderr, and golden/harness code captures only printer bytes.
package log

import (
	"context"
	"log/slog"
	"os"
)

// defaultLogger backs every package-level call. It is slog's default until
// SetDefault installs a compiler-wide handler.
var defaultLogger = slog.New(slog.NewTextHandler(os.Stderr, nil))

// ctxKey carries a context-scoped logger installed by WithGroup.
type ctxKey struct{}

// SetDefault installs h as the handler behind every package-level log call.
// Pass nil to restore the stderr text handler.
func SetDefault(h slog.Handler) {
	if h == nil {
		defaultLogger = slog.New(slog.NewTextHandler(os.Stderr, nil))
		return
	}
	defaultLogger = slog.New(h)
}

// loggerFrom returns the context-scoped logger installed by WithGroup, or
// the package default when the context carries none.
func loggerFrom(ctx context.Context) *slog.Logger {
	if l, ok := ctx.Value(ctxKey{}).(*slog.Logger); ok && l != nil {
		return l
	}
	return defaultLogger
}

// Debug logs msg at debug level through the context's logger.
func Debug(ctx context.Context, msg string, args ...any) {
	loggerFrom(ctx).Log(ctx, slog.LevelDebug, msg, args...)
}

// Info logs msg at info level through the context's logger.
func Info(ctx context.Context, msg string, args ...any) {
	loggerFrom(ctx).Log(ctx, slog.LevelInfo, msg, args...)
}

// Warn logs msg at warn level through the context's logger.
func Warn(ctx context.Context, msg string, args ...any) {
	loggerFrom(ctx).Log(ctx, slog.LevelWarn, msg, args...)
}

// Error logs msg at error level through the context's logger.
func Error(ctx context.Context, msg string, args ...any) {
	loggerFrom(ctx).Log(ctx, slog.LevelError, msg, args...)
}

// WithGroup returns a context whose logger groups subsequent records under
// name (slog.Group support for per-pass / per-node records). Extra attrs
// are attached to the group. Typical use:
//
//	ctx = log.WithGroup(ctx, "pass", slog.String("id", p.ID()))
//	log.Info(ctx, "running", slog.Int("nodes", n))
func WithGroup(ctx context.Context, name string, attrs ...any) context.Context {
	l := loggerFrom(ctx).WithGroup(name)
	if len(attrs) > 0 {
		l = l.With(attrs...)
	}
	return context.WithValue(ctx, ctxKey{}, l)
}
