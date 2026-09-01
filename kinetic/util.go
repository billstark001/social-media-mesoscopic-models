package kinetic

import (
	"strings"
)

func normalize(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}
