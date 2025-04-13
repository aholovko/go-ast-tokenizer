package main

import (
	_ "embed"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	//go:embed testdata/captLocal.txt
	captLocal string
	//go:embed testdata/commentFormatting.txt
	commentFormatting string
)

func TestStyleChecker(t *testing.T) {
	tests := []struct {
		name    string
		snippet string
		want    WarningMask
	}{
		{
			name:    "captLocal",
			snippet: captLocal,
			want:    CaptLocal,
		},
		{
			name:    "commentFormatting",
			snippet: commentFormatting,
			want:    CommentFormatting,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := checker.Check(tc.snippet)
			assert.NoError(t, err)
			assert.Equal(t, tc.want, got&tc.want, "expected warning mask to include %v", tc.want)
		})
	}
}
