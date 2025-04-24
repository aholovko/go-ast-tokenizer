package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strings"
)

// Token represents a lexeme extracted from Go source with its type and position.
type Token struct {
	Text string
	Kind token.Token
	Pos  token.Position
}

// Tokenizer parses Go source and returns tokens in source order.
type Tokenizer struct {
	fset *token.FileSet
}

// specialTags maps Go token.Tokens to custom special-token strings.
var specialTags = map[token.Token]string{
	token.IDENT:   "<IDENT>",
	token.INT:     "<LIT_INT>",
	token.FLOAT:   "<LIT_FLOAT>",
	token.IMAG:    "<LIT_IMAG>",
	token.CHAR:    "<LIT_CHAR>",
	token.STRING:  "<LIT_STRING>",
	token.COMMENT: "<COMMENT>",

	// assignment operators
	token.ASSIGN:     "<ASSIGN_OP>",
	token.DEFINE:     "<ASSIGN_OP>",
	token.ADD_ASSIGN: "<ASSIGN_OP>",
	token.SUB_ASSIGN: "<ASSIGN_OP>",
	token.MUL_ASSIGN: "<ASSIGN_OP>",
	token.QUO_ASSIGN: "<ASSIGN_OP>",
	token.REM_ASSIGN: "<ASSIGN_OP>",

	// binary operators
	token.ADD:     "<BINARY_OP>",
	token.SUB:     "<BINARY_OP>",
	token.MUL:     "<BINARY_OP>",
	token.QUO:     "<BINARY_OP>",
	token.REM:     "<BINARY_OP>",
	token.AND:     "<BINARY_OP>",
	token.AND_NOT: "<BINARY_OP>",
	token.OR:      "<BINARY_OP>",
	token.XOR:     "<BINARY_OP>",
	token.SHL:     "<BINARY_OP>",
	token.SHR:     "<BINARY_OP>",

	// keywords & punctuation
	token.IF:     "<IF>",
	token.ELSE:   "<ELSE>",
	token.SWITCH: "<SWITCH>",
	token.FUNC:   "<FUNC>",
	token.CASE:   "<CASE>",
	token.LBRACE: "<LBRACE>",
	token.LPAREN: "<LPAREN>",
	token.RBRACE: "<RBRACE>",
	token.RPAREN: "<RPAREN>",
	token.COLON:  "<COLON>",
}

// NewTokenizer returns a new Tokenizer.
func NewTokenizer() *Tokenizer {
	return &Tokenizer{fset: token.NewFileSet()}
}

// TokenizedString returns a space-separated sequence of special-token strings.
func (t *Tokenizer) TokenizedString(src string) (string, error) {
	tokens, err := t.tokenize(src)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	for i, tok := range tokens {
		if tag, ok := specialTags[tok.Kind]; ok {
			switch tok.Kind {
			case token.IDENT, token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING:
				sb.WriteString(tag)
				sb.WriteByte(' ')
				sb.WriteString(tok.Text)
			default:
				sb.WriteString(tag)
			}
		} else {
			sb.WriteString(tok.Text)
		}
		if i < len(tokens)-1 {
			sb.WriteByte(' ')
		}
	}
	return sb.String(), nil
}

// tokenize parses src and returns all tokens in the order they appear.
func (t *Tokenizer) tokenize(src string) ([]Token, error) {
	file, err := parser.ParseFile(t.fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var tokens []Token

	t.collectComments(file, &tokens)
	t.collectAST(file, &tokens)

	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].Pos.Offset < tokens[j].Pos.Offset
	})

	return tokens, nil
}

// collectComments adds comment tokens to tokens.
func (t *Tokenizer) collectComments(file *ast.File, tokens *[]Token) {
	for _, cg := range file.Comments {
		for _, c := range cg.List {
			t.addToken(tokens, token.COMMENT, c.Text, c.Slash)
		}
	}
}

// collectAST walks the AST and adds relevant tokens to tokens.
func (t *Tokenizer) collectAST(file *ast.File, tokens *[]Token) {
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Ident:
			t.addToken(tokens, token.IDENT, x.Name, x.NamePos)
		case *ast.BasicLit:
			t.addToken(tokens, x.Kind, x.Value, x.ValuePos)
		case *ast.AssignStmt:
			t.addToken(tokens, x.Tok, x.Tok.String(), x.TokPos)
		case *ast.BinaryExpr:
			t.addToken(tokens, x.Op, x.Op.String(), x.OpPos)
		case *ast.IfStmt:
			t.addToken(tokens, token.IF, "if", x.If)
			if x.Else != nil {
				t.addToken(tokens, token.ELSE, "else", x.Else.Pos())
			}
		case *ast.SwitchStmt:
			t.addToken(tokens, token.SWITCH, "switch", x.Switch)
		case *ast.FuncType:
			t.addToken(tokens, token.FUNC, "func", x.Func)
		case *ast.CaseClause:
			if x.Case.IsValid() {
				t.addToken(tokens, token.CASE, "case", x.Case)
			}
			t.addToken(tokens, token.COLON, ":", x.Colon)
		case *ast.BlockStmt:
			t.addToken(tokens, token.LBRACE, "{", x.Lbrace)
			t.addToken(tokens, token.RBRACE, "}", x.Rbrace)
		case *ast.ParenExpr:
			t.addToken(tokens, token.LPAREN, "(", x.Lparen)
			t.addToken(tokens, token.RPAREN, ")", x.Rparen)
		}
		return true
	})
}

// addToken appends a new token to tokens.
func (t *Tokenizer) addToken(tokens *[]Token, kind token.Token, text string, pos token.Pos) {
	*tokens = append(*tokens, Token{
		Text: text,
		Kind: kind,
		Pos:  t.fset.Position(pos),
	})
}
