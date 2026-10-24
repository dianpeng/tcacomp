package lexer

// FileSet is the loader-owned set of source files (design Task 5's
// ParseModule(fset, path) signature; mirrors go/token.FileSet's role). Files
// are append-only and identified by their index, so a Pos's *File pointer
// always refers to a stable, loader-managed object.
type FileSet struct {
	files []*File
}

// NewFileSet creates an empty file set.
func NewFileSet() *FileSet { return &FileSet{} }

// AddFile registers one source file (name is the path as given to the loader)
// and returns the *File scanners and parsers attribute positions to.
func (s *FileSet) AddFile(name, src string) *File {
	f := &File{Name: name, Src: src}
	s.files = append(s.files, f)
	return f
}

// Files returns the registered files in registration order.
func (s *FileSet) Files() []*File { return s.files }
