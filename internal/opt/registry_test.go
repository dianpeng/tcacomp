package opt_test

import (
	"strings"
	"testing"

	"tcacomp/internal/opt"
	"tcacomp/internal/opt/schedule"
)

// The registry pins (quick-261018-l36 task 1). The registry is PROCESS-GLOBAL:
// these tests never assert exact pipeline membership or absolute length —
// only relative order and membership of specific IDs. Fake IDs and priorities
// are chosen to never collide with the real scale anchors (100 dce, 900
// schedule).

// The schedule import below exists for its init() self-registration; this
// assertion additionally pins that the registered factory's product type is
// the exported Scheduler.
var _ opt.Pass = schedule.Scheduler{}

// mustRegister registers spec, failing the test on panic with the message.
func mustRegister(t *testing.T, spec opt.PassSpec) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Register(%q) panicked: %v", spec.ID, r)
		}
	}()
	opt.Register(spec)
}

// registryIndex returns the index of id in the Registered() list, or -1.
func registryIndex(id string) int {
	for i, spec := range opt.Registered() {
		if spec.ID == id {
			return i
		}
	}
	return -1
}

// pipelineIndex returns the index of the pass with the given ID in ps, or -1.
func pipelineIndex(ps []opt.Pass, id string) int {
	for i, p := range ps {
		if p.ID() == id {
			return i
		}
	}
	return -1
}

func TestRegistryScheduleRegisteredByImport(t *testing.T) {
	spec, ok := opt.Lookup("schedule")
	if !ok {
		t.Fatal("Lookup(schedule) not found — the schedule import must self-register")
	}
	if spec.Priority != 900 {
		t.Errorf("schedule priority = %d, want 900", spec.Priority)
	}
	if spec.Factory == nil {
		t.Fatal("schedule spec has a nil Factory")
	}
	if got := spec.Factory(); got.ID() != "schedule" {
		t.Errorf("schedule factory produces pass %q, want %q", got.ID(), "schedule")
	}
	if _, ok := opt.Lookup("no-such-pass"); ok {
		t.Fatal("Lookup(no-such-pass) found a spec that was never registered")
	}
}

func TestRegistryPriorityOrderStable(t *testing.T) {
	// Distinct priorities order ascending; equal priorities keep
	// registration order. Priorities 400/500 stay away from 100/900.
	mustRegister(t, opt.PassSpec{ID: "l36-fake-a", Priority: 500, Factory: func() opt.Pass { return &stubPass{id: "l36-fake-a"} }})
	mustRegister(t, opt.PassSpec{ID: "l36-fake-b", Priority: 400, Factory: func() opt.Pass { return &stubPass{id: "l36-fake-b"} }})
	mustRegister(t, opt.PassSpec{ID: "l36-fake-c", Priority: 500, Factory: func() opt.Pass { return &stubPass{id: "l36-fake-c"} }})

	ia, ib, ic := registryIndex("l36-fake-a"), registryIndex("l36-fake-b"), registryIndex("l36-fake-c")
	if ib < 0 || ia < 0 || ic < 0 {
		t.Fatalf("fake specs missing from Registered(): a=%d b=%d c=%d", ia, ib, ic)
	}
	if !(ib < ia && ia < ic) {
		t.Errorf("Registered() order wrong: want b(400) < a(500, first) < c(500, second), got b=%d a=%d c=%d", ib, ia, ic)
	}
	// Real anchors stay ordered against the fakes: everything < schedule.
	if isch := registryIndex("schedule"); isch < ic {
		t.Errorf("schedule (%d) must sort after every smaller-priority fake (c at %d)", isch, ic)
	}
}

func TestRegistryPanics(t *testing.T) {
	expectPanic := func(name, wantSub string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: Register did not panic", name)
			}
			msg, ok := r.(string)
			if !ok {
				t.Fatalf("%s: panic value is %T (%v), want string", name, r, r)
			}
			if !strings.Contains(msg, wantSub) {
				t.Fatalf("%s: panic %q does not mention %q", name, msg, wantSub)
			}
		}()
		fn()
	}

	expectPanic("empty ID", `empty pass ID`, func() {
		opt.Register(opt.PassSpec{Factory: func() opt.Pass { return &stubPass{id: "x"} }})
	})
	expectPanic("nil Factory", `nil Factory`, func() {
		opt.Register(opt.PassSpec{ID: "l36-fake-nil"})
	})
	expectPanic("duplicate ID", `l36-fake-a`, func() {
		opt.Register(opt.PassSpec{ID: "l36-fake-a", Priority: 500, Factory: func() opt.Pass { return &stubPass{id: "l36-fake-a"} }})
	})
	expectPanic("factory-ID mismatch", `l36-fake-wrong`, func() {
		opt.Register(opt.PassSpec{ID: "l36-fake-wrong", Factory: func() opt.Pass { return &stubPass{id: "l36-fake-other"} }})
	})
}

func TestBuildPipelineFreshInstancesInRegisteredOrder(t *testing.T) {
	first := opt.BuildPipeline()
	second := opt.BuildPipeline()

	// Every pipeline instance is fresh: no fake instance is shared across two
	// BuildPipeline calls (factories construct, they do not memoize). The
	// check is scoped to the pointer-returning fake factories — a stateless
	// value-type pass (Scheduler{}) is trivially equal to its own fresh copy,
	// and interface equality cannot distinguish construction from sharing.
	fakeIDs := map[string]bool{"l36-fake-a": true, "l36-fake-b": true, "l36-fake-c": true}
	seen := make(map[opt.Pass]bool)
	for _, p := range first {
		if fakeIDs[p.ID()] {
			seen[p] = true
		}
	}
	for _, p := range second {
		if fakeIDs[p.ID()] && seen[p] {
			t.Fatalf("BuildPipeline returned a shared instance for pass %q", p.ID())
		}
	}

	// Pipeline order mirrors Registered() order: for the fake trio plus the
	// real anchors, relative positions agree between the two listings.
	pipeline := opt.BuildPipeline()
	specs := opt.Registered()
	for _, id := range []string{"l36-fake-b", "l36-fake-a", "l36-fake-c", "schedule"} {
		ri, pi := registryIndex(id), pipelineIndex(pipeline, id)
		if ri < 0 || pi < 0 {
			t.Fatalf("pass %q missing (registry=%d pipeline=%d)", id, ri, pi)
		}
		if len(specs) <= ri || specs[ri].ID != id {
			t.Fatalf("Registered()[%d] = %q, want %q", ri, specs[ri].ID, id)
		}
	}
	// Global relative order: registry order and pipeline order agree pairwise.
	order := []string{"l36-fake-b", "l36-fake-a", "l36-fake-c", "schedule"}
	for i := 1; i < len(order); i++ {
		if !(pipelineIndex(pipeline, order[i-1]) < pipelineIndex(pipeline, order[i])) {
			t.Errorf("pipeline order wrong: %s must precede %s", order[i-1], order[i])
		}
	}
}
