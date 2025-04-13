// go build -buildmode=c-shared -o _checkstyle.so
package main

import "C"

var checker *StyleChecker

func init() {
	var err error

	checker, err = NewStyleChecker()
	if err != nil {
		panic(err)
	}
}

//export check
func check(src *C.char) C.int {
	snippet := C.GoString(src)

	if len(snippet) == 0 {
		return C.int(-1)
	}

	warnings, err := checker.Check(snippet)
	if err != nil {
		return C.int(-1)
	}

	return C.int(warnings)
}

func main() {}
