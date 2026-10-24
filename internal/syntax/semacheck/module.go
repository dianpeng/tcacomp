package semacheck

import (
	"tcacomp/internal/diagnostic"
	"tcacomp/internal/syntax/ast"
)

// Module-scope checks (design §4.1): rules 1 and 3-4. Include cycles and
// missing files are the module loader's mechanics (it reports them while
// resolving); this function validates what is visible on the merged module
// tree.
func CheckModule(m *ast.Module) []diagnostic.Diagnostic {
	var diags []diagnostic.Diagnostic

	// Rule 1: duplicate kernel names across the merged table.
	seen := map[string]bool{}
	for _, k := range m.Kernels {
		if seen[k.Name] {
			diags = append(diags, diagnostic.Diagnostic{
				Severity: diagnostic.Error,
				Span:     spanOf(k.Pos()),
				Message:  "duplicate kernel name " + k.Name + " (rule 1)",
			})
		}
		seen[k.Name] = true
	}

	// Rule 3: @runtime present → cxx_path must be present and a string.
	if m.Runtime != nil {
		var cxxPath *ast.KeyedConst
		for _, a := range m.Runtime.Args {
			if a.Key == "cxx_path" {
				cxxPath = a
			} else {
				diags = append(diags, diagnostic.Diagnostic{
					Severity: diagnostic.Error,
					Span:     spanOf(a.Pos()),
					Message:  "unknown @runtime key " + a.Key + " (want cxx_path, rule 3)",
				})
			}
		}
		if cxxPath == nil {
			diags = append(diags, diagnostic.Diagnostic{
				Severity: diagnostic.Error,
				Span:     spanOf(m.Runtime.Pos()),
				Message:  "@runtime requires cxx_path=\"...\" (rule 3)",
			})
		} else if cxxPath.Value.Kind != ast.ConstString {
			diags = append(diags, diagnostic.Diagnostic{
				Severity: diagnostic.Error,
				Span:     spanOf(cxxPath.Pos()),
				Message:  "@runtime cxx_path must be a string literal (rule 3)",
			})
		}
	}

	// Rule 4: annotation registry — v0 knows exactly @name{cxx=string}.
	for _, k := range m.Kernels {
		for _, ann := range k.Anns {
			switch ann.Name {
			case "name":
				if len(ann.Args) != 1 || ann.Args[0].Key != "cxx" {
					diags = append(diags, diagnostic.Diagnostic{
						Severity: diagnostic.Error,
						Span:     spanOf(ann.Pos()),
						Message:  "@name accepts exactly {cxx=string} (rule 4)",
					})
					continue
				}
				if ann.Args[0].Value.Kind != ast.ConstString {
					diags = append(diags, diagnostic.Diagnostic{
						Severity: diagnostic.Error,
						Span:     spanOf(ann.Args[0].Pos()),
						Message:  "@name cxx must be a string literal (rule 4)",
					})
				}
			default:
				diags = append(diags, diagnostic.Diagnostic{
					Severity: diagnostic.Error,
					Span:     spanOf(ann.Pos()),
					Message:  "unknown annotation @" + ann.Name + " (rule 4: the v0 registry holds only @name{cxx})",
				})
			}
		}
	}
	return diags
}
