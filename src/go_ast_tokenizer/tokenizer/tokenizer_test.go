package main

import (
	_ "embed"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	//go:embed testdata/binary_expression.go
	binaryExpression string
	//go:embed testdata/comment.go
	comment string
	//go:embed testdata/identifier.go
	identifier string
	//go:embed testdata/integer_literal.go
	integerLiteral string
	//go:embed testdata/parentheses.go
	parentheses string
	//go:embed testdata/simple_assignment.go
	simpleAssignment string
	//go:embed testdata/string_literal.go
	stringLiteral string
)

func TestTokenizedString(t *testing.T) {
	tok := NewTokenizer()

	tests := []struct {
		name string
		code string
		want string
	}{
		{name: "binary expression", code: binaryExpression, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> x <IDENT> y <IDENT> int <IDENT> _ <ASSIGN_OP> <IDENT> x <BINARY_OP> <IDENT> y <RBRACE>"},
		{name: "comment", code: comment, want: "<COMMENT> <IDENT> main"},
		{name: "identifier", code: identifier, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> foo <IDENT> int <IDENT> _ <ASSIGN_OP> <IDENT> foo <RBRACE>"},
		{name: "integer literal", code: integerLiteral, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> _ <ASSIGN_OP> <LIT_INT> 42 <RBRACE>"},
		{name: "parentheses", code: parentheses, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> _ <ASSIGN_OP> <LPAREN> <LIT_INT> 40 <BINARY_OP> <LIT_INT> 2 <RPAREN> <BINARY_OP> <LIT_INT> 2 <RBRACE>"},
		{name: "simple assignment", code: simpleAssignment, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> a <IDENT> int <IDENT> b <ASSIGN_OP> <IDENT> a <IDENT> _ <ASSIGN_OP> <IDENT> b <RBRACE>"},
		{name: "string literal", code: stringLiteral, want: "<IDENT> main <FUNC> <IDENT> main <LBRACE> <IDENT> _ <ASSIGN_OP> <LIT_STRING> \"hello\" <RBRACE>"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tokenized, err := tok.TokenizedString(tc.code)
			assert.NoError(t, err)
			assert.Equal(t, tc.want, tokenized)
		})
	}
}
