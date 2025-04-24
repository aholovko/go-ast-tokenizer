// go build -buildmode=c-shared -o _tokenizer.so
package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

var tokenizer *Tokenizer

func init() {
	tokenizer = NewTokenizer()
}

//export tokenize
func tokenize(src *C.char) *C.char {
	snippet := C.GoString(src)
	if len(snippet) == 0 {
		return nil
	}

	// Generate the tokenized string
	tokStr, err := tokenizer.TokenizedString(snippet)
	if err != nil {
		return nil
	}

	// Convert Go string to C string
	cstr := C.CString(tokStr)
	return cstr
}

//export freeCString
func freeCString(cstr *C.char) {
	C.free(unsafe.Pointer(cstr))
}

func main() {}
