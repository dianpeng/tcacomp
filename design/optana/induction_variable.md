# Induction Variable

- Induction Variable
  - 1) They are value objects, and its value can be compared
    - Not only by its name, but also need to compare its value, ie identity check

  - 2) To allow more optimization opportunity
    - We define following relationship between Induction Variable
    - Induction Variable forms the basic condition for loop optimization
      - Loop merging
      - Loop fusion
      - Loop combine (not introduced yet)

- Bounded Induction Variable
  - When induction variable is applied in map operators, it turns into **Bounded Induction Variable**

```
dimvar @m; // induction variable
map(%x[@m/M], ... // <- @m becomes bounded induction variable, ie it has an upper bound
                 // the syntax x[@m/M] becomes a bounded induction variable, which is
                 // a tuple as: (<dimvar's tuple>, Max)
                 // Max is the upper bound of the dimvar binding


```
  - Why this design ?
    - dimvar is just a loop induction variable
      - it does not 1:1 mapping to a loop
    - we will allow user to visit the loop in an arbitary fashion

```

// NOTES, this is not supported in this verison, but will be supported
//
// It defines the access of y will requires offset from the @m. This is useful for
// convlutional network commonly. This also means the the iteration CAN BE an iterator
map(%x[@m], %y[@m + @n]) ...

// possible lower, notes this is trivial which we should NOT lower to this, since this
// prohibits us from performing several types of optimization. Better naturalize it
for (auto @m = 0; @m < M; @m++) {
  for (auto @n = @m + 0; @n < N; @n++) { // <- this becomes an irregular loop
  }
}

// a normalized lower
//
// this requires an **analysis pass** called range inference to decide the 0 starting
// maximum bound for BoundedInductionVariable @n based on @m. Will document somewhere
for (auto @m = 0; @m < M; @m++) {
  for (auto @n = 0; @n < N - M; @n++) {
  }
}

````

  - So introducing the BoundedInductionVariable will make the backend optimization framework
    much flexible for future new syntax feature supports
    - Making sure to understand the mapping
      - 1) induction variable means an UNBOUNDED loop induction variable definition
      - 2) bounded induction variable is generated implicitly whenever a binding between iv to dimension happened, which learns the maximum value
      - 3) for now, we do not support arbitary access index function
        - BUT we will support it, which means we need to perform range inference analysis pass to decide a normalized (0 based) induction variable upper bound afterwards

- Induction Variable(IV) Relationship
  - Defined between **induction variable**
  - EQ/NE Relationship
    - 2 induction variables are equal when they generate **same iteration space in same way**
      - induction variables are (init, step)
      - 2 induction variables are equal when its 2 tuples are equal
    - Algebric Rule
      - a = b => b = a
      - a = b => !(a != b)

- Bounded Induction Variable(BIV) Relationship
  - Defined between **induction variable**
  - EQ/NE Relationship
    - 2 IV are equal when: 1) corresponding dimvars are same; 2) upper bound Max are the same
    - Algebric rule is same as IV correspondingly
  - Include/Belong Relationship
    - BIV: A include BIV: B 
      - B's iteration space is **subset of** A's defined iteration space
      - For most of case, we are dealing with noramlized induction variable
        - init = 0; step = 1
      - It means A's Max <= B's Max
      - Denoted as A ~> B // A includes B
    - Notes the include relationship is not commutative
    - Belong is the reverse
      - A include B => B belongs to A
      - A ~> B      => B <~ A
    - Notes this relationship is not used in optimizer, but will use for partial loop fusion
      - A loop nests that are **belonged to another loop nests B** can be partial merge
        - Merged with guard

- Notes, **ordered set** of IV or BIV's relationship can be defined accordingly
  - For a set IVs
    - EQ => all IVs in sets are equaled to another set
    - NE => NOT eq
  - For a set of BIVs
    - EQ => all IVs in sets are equaled to another set
    - NE => NOT eq
    - INCLDUES => ALL BIVs in one set include the other corresponding one
    - BELONGS  => ALL BIVs in one set include the other corresponding one
