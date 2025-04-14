package main

import (
	"fmt"
	"go/parser"
	"go/token"
	"go/types"
	"log/slog"
	"os"
	"runtime"
	"strings"

	"github.com/go-critic/go-critic/checkers"
	"github.com/go-critic/go-critic/linter"
)

// init configures slog to log messages to a file ("warnings.jsonl").
func init() {
	logFile, err := os.OpenFile("warnings.jsonl", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		slog.Error("failed to open log file", "err", err)
		return
	}

	handler := slog.NewJSONHandler(logFile, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})

	slog.SetDefault(slog.New(handler))
}

// WarningMask is a bitmask encoding violated style rules.
type WarningMask int

const (
	// NoWarnings indicates that no warnings were detected.
	NoWarnings WarningMask = 0

	// AssignOp detects assignments that can be simplified by using assignment operators.
	AssignOp WarningMask = 1 << iota
	// BuiltinShadow detects when predeclared identifiers are shadowed in assignments.
	BuiltinShadow
	// CaptLocal detects capitalized names for local variables.
	CaptLocal
	// CommentFormatting detects comments with non-idiomatic formatting.
	CommentFormatting
	// DefaultCaseOrder detects when default case in switch isn't on 1st or last position.
	DefaultCaseOrder
	// Elseif detects else with nested if statement that can be replaced with else-if.
	Elseif
	// IfElseChain detects repeated if-else statements and suggests to replace them with switch statement.
	IfElseChain
	// ImportShadow detects when imported package names shadowed in the assignments.
	ImportShadow
	// NewDeref detects immediate dereferencing of new expressions.
	NewDeref
	// ParamTypeCombine detects if function parameters could be combined by type and suggest the way to do it.
	ParamTypeCombine
	// RegexpMust detects regexp.Compile* that can be replaced with regexp.MustCompile*.
	RegexpMust
	// SingleCaseSwitch detects switch statements that could be better written as if statement.
	SingleCaseSwitch
	// SwitchTrue detects switch-over-bool statements that use explicit true tag value.
	SwitchTrue
	// TypeSwitchVar detects type switches that can benefit from type guard clause with variable.
	TypeSwitchVar
	// TypeUnparen detects unneeded parenthesis inside type expressions and suggests to remove them.
	TypeUnparen
	// Underef detects dereference expressions that can be omitted.
	Underef
	// Unlambda detects function literals that can be simplified.
	Unlambda
	// Unslice detects slice expressions that can be simplified to sliced expression itself.
	Unslice
	// ValSwap detects value swapping code that are not using parallel assignment.
	ValSwap
	// WrapperFunc detects function calls that can be replaced with convenience wrappers.
	WrapperFunc
)

// StyleChecker analyzes a Go code snippet and flags violations of go-critic's standard style rules.
type StyleChecker struct {
	rules []*styleRule
}

type styleRule struct {
	name    string
	checker *linter.Checker
	mask    WarningMask
}

// NewStyleChecker initializes and returns a new StyleChecker instance.
func NewStyleChecker() (*StyleChecker, error) {
	if err := checkers.InitEmbeddedRules(); err != nil {
		return nil, fmt.Errorf("failed to init embedded rules: %w", err)
	}

	var checkersInfo []*linter.CheckerInfo
	for _, ci := range linter.GetCheckersInfo() {
		if hasTag(ci.Tags, "style") && !hasTag(ci.Tags, "experimental") {
			checkersInfo = append(checkersInfo, ci)
		}
	}

	if len(checkersInfo) == 0 {
		return nil, fmt.Errorf("no style checkers found")
	}

	fset := token.NewFileSet()
	sizes := types.SizesFor("gc", runtime.GOARCH)
	ctx := linter.NewContext(fset, sizes)

	var rules []*styleRule

	mapping := map[string]WarningMask{
		"assignOp":          AssignOp,
		"builtinShadow":     BuiltinShadow,
		"captLocal":         CaptLocal,
		"commentFormatting": CommentFormatting,
		"defaultCaseOrder":  DefaultCaseOrder,
		"elseif":            Elseif,
		"ifElseChain":       IfElseChain,
		"importShadow":      ImportShadow,
		"newDeref":          NewDeref,
		"paramTypeCombine":  ParamTypeCombine,
		"regexpMust":        RegexpMust,
		"singleCaseSwitch":  SingleCaseSwitch,
		"switchTrue":        SwitchTrue,
		"typeSwitchVar":     TypeSwitchVar,
		"typeUnparen":       TypeUnparen,
		"underef":           Underef,
		"unlambda":          Unlambda,
		"unslice":           Unslice,
		"valSwap":           ValSwap,
		"wrapperFunc":       WrapperFunc,
	}

	for _, info := range checkersInfo {
		c, err := linter.NewChecker(ctx, info)
		if err != nil {
			return nil, fmt.Errorf("failed to create linter %q: %w", info.Name, err)
		}

		mask, ok := mapping[info.Name]
		if !ok {
			slog.Warn("unknown checker", "checker", info.Name)
		}

		rules = append(rules, &styleRule{name: c.Info.Name, checker: c, mask: mask})
	}

	return &StyleChecker{
		rules: rules,
	}, nil
}

func hasTag(tags []string, substr string) bool {
	for _, tag := range tags {
		if strings.Contains(tag, substr) {
			return true
		}
	}
	return false
}

// Check runs the style checkers on the provided Go code snippet and returns the combined warning mask.
func (sc *StyleChecker) Check(snippet string) (WarningMask, error) {
	fset := token.NewFileSet()

	ast, parseErr := parser.ParseFile(fset, "", snippet, parser.ParseComments)
	if parseErr != nil {
		return NoWarnings, parseErr
	}

	mask := NoWarnings

	for _, r := range sc.rules {
		var warnings []linter.Warning

		err := func() (err error) {
			defer func() {
				if rec := recover(); rec != nil {
					slog.Error("checker panicked", "checker", r.name, "panic", rec)
					err = fmt.Errorf("checker %q panicked with: %v", r.name, rec)
				}
			}()

			warnings = r.checker.Check(ast)
			return nil
		}()

		if err != nil {
			return NoWarnings, err
		}

		if len(warnings) > 0 {
			mask |= r.mask

			for _, warning := range warnings {
				slog.Info(r.name, "warning", warning.Text)
			}
		}
	}

	return mask, nil
}
