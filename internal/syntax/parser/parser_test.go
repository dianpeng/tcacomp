package parser

import "tcacomp/internal/syntax/lexer"

// testFile builds a lexer.File for a test source (helper for the parser
// tests in this package).
func testFile(name, src string) *lexer.File {
	return (&lexer.FileSet{}).AddFile(name, src)
}
