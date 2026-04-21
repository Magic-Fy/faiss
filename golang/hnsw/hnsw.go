// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Package hnsw provides a pure-Go implementation of the Hierarchical
// Navigable Small World (HNSW) approximate nearest-neighbor index.
//
// Reference: Yu. A. Malkov, D. A. Yashunin, "Efficient and robust
// approximate nearest neighbor search using Hierarchical Navigable
// Small World graphs", arXiv 2016.
package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
)

// MetricType selects the distance function used by the index.
type MetricType int

const (
	// MetricL2 uses squared Euclidean (L2) distance.
	MetricL2 MetricType = iota
	// MetricInnerProduct uses negative inner product (so that
	// "closest" still means smallest value).
	MetricInnerProduct
)

// ---------------------------------------------------------------------------
// Heap helpers
// ---------------------------------------------------------------------------

// nodeDist is a (distance, id) pair used in priority queues.
type nodeDist struct {
	dist float32
	id   int32
}

// maxHeap is a max-heap on distance (top = farthest).
// Used to maintain the result set: we keep the k nearest, so we
// evict the farthest when the heap overflows.
type maxHeap []nodeDist

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(nodeDist)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// minHeap is a min-heap on distance (top = nearest).
// Used for the candidate queue during search/construction.
type minHeap []nodeDist

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(nodeDist)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// ---------------------------------------------------------------------------
// HNSW index
// ---------------------------------------------------------------------------

// HNSW is an approximate nearest-neighbor index based on Hierarchical
// Navigable Small World graphs.
type HNSW struct {
	// Dimensionality of stored vectors.
	d int

	// Distance metric.
	metric MetricType

	// Base number of bidirectional links per node (level > 0 uses M,
	// level 0 uses 2*M).
	m int

	// efConstruction is the size of the dynamic candidate list during
	// graph construction.
	EfConstruction int

	// EfSearch is the size of the dynamic candidate list during search.
	EfSearch int

	// assignProbas[l] is the probability that a new node is assigned
	// to at most level l.  len == max possible levels.
	assignProbas []float64

	// cumNNeighborPerLevel[l] is the cumulative neighbor count up to
	// (excluding) level l.  len == len(assignProbas)+1.
	cumNNeighborPerLevel []int

	// levels[i] stores the number of layers node i participates in
	// (= assigned_level + 1).  A node at level 0 has levels[i]==1.
	levels []int

	// offsets[i] is the start index in neighbors[] for node i.
	// len == ntotal+1, offsets[ntotal] == len(neighbors).
	offsets []int

	// neighbors is a flat array of neighbor IDs.  -1 means "empty slot".
	neighbors []int32

	// entryPoint is the ID of the current graph entry point (-1 if empty).
	entryPoint int32

	// maxLevel is the level of the entry point.
	maxLevel int

	// vectors stores the raw float data, row-major: vectors[i*d:(i+1)*d].
	vectors []float32

	// rng is the random source for level assignment.
	rng *rand.Rand
}

// New creates an empty HNSW index.
//
//   - d             : vector dimension
//   - m             : number of bidirectional links per node (recommended 16–64)
//   - efConstruction: candidate list size during construction (recommended ≥ 2*m)
//   - metric        : MetricL2 or MetricInnerProduct
func New(d, m, efConstruction int, metric MetricType) *HNSW {
	h := &HNSW{
		d:              d,
		metric:         metric,
		m:              m,
		EfConstruction: efConstruction,
		EfSearch:       16,
		entryPoint:     -1,
		maxLevel:       -1,
		rng:            rand.New(rand.NewSource(12345)),
	}
	h.offsets = []int{0}
	h.setDefaultProbas(m, 1.0/math.Log(float64(m)))
	return h
}

// Ntotal returns the number of vectors currently stored in the index.
func (h *HNSW) Ntotal() int { return len(h.levels) }

// setDefaultProbas initialises the level probabilities following the
// original HNSW paper: P(level==l) ∝ exp(-l/levelMult).
func (h *HNSW) setDefaultProbas(m int, levelMult float64) {
	h.cumNNeighborPerLevel = []int{0}
	nn := 0
	for level := 0; ; level++ {
		proba := math.Exp(-float64(level)/levelMult) * (1 - math.Exp(-1/levelMult))
		if proba < 1e-9 {
			break
		}
		h.assignProbas = append(h.assignProbas, proba)
		if level == 0 {
			nn += m * 2
		} else {
			nn += m
		}
		h.cumNNeighborPerLevel = append(h.cumNNeighborPerLevel, nn)
	}
}

// nbNeighbors returns the maximum number of neighbors a node can have
// at the given level.
func (h *HNSW) nbNeighbors(level int) int {
	return h.cumNNeighborPerLevel[level+1] - h.cumNNeighborPerLevel[level]
}

// cumNbNeighbors returns the cumulative neighbor count up to (excluding)
// level.
func (h *HNSW) cumNbNeighbors(level int) int {
	return h.cumNNeighborPerLevel[level]
}

// neighborRange returns [begin, end) of the neighbors slice for node no
// at the given level.
func (h *HNSW) neighborRange(no int, level int) (begin, end int) {
	o := h.offsets[no]
	begin = o + h.cumNbNeighbors(level)
	end = o + h.cumNbNeighbors(level+1)
	return
}

// randomLevel picks a level for a new node using the configured
// probability distribution.
func (h *HNSW) randomLevel() int {
	f := h.rng.Float64()
	for level, p := range h.assignProbas {
		if f < p {
			return level
		}
		f -= p
	}
	return len(h.assignProbas) - 1
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

// dist computes the distance between the query vector q and stored vector id.
func (h *HNSW) dist(q []float32, id int32) float32 {
	v := h.vectors[int(id)*h.d : int(id)*h.d+h.d]
	switch h.metric {
	case MetricInnerProduct:
		return -innerProduct(q, v)
	default:
		return l2Squared(q, v)
	}
}

// symmetricDist computes the distance between two stored vectors.
func (h *HNSW) symmetricDist(a, b int32) float32 {
	va := h.vectors[int(a)*h.d : int(a)*h.d+h.d]
	return h.dist(va, b)
}

func l2Squared(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func innerProduct(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

// shrinkNeighborList prunes the candidate set so that it contains at most
// maxSize neighbors, keeping the "good" ones according to the HNSW
// heuristic: a candidate v1 is kept only if no already-selected neighbor
// v2 is closer to v1 than the query is.
//
// input is a max-heap (farthest on top); output is built in nearest-first
// order.
func (h *HNSW) shrinkNeighborList(
	q []float32,
	input *maxHeap,
	maxSize int,
) []nodeDist {
	// Convert max-heap → min-heap to process nearest first.
	tmp := make(minHeap, len(*input))
	for i, nd := range *input {
		tmp[i] = nd
	}
	heap.Init(&tmp)

	var output []nodeDist
	for tmp.Len() > 0 && len(output) < maxSize {
		v1 := heap.Pop(&tmp).(nodeDist)
		good := true
		for _, v2 := range output {
			distV1V2 := h.symmetricDist(v2.id, v1.id)
			if distV1V2 < v1.dist {
				good = false
				break
			}
		}
		if good {
			output = append(output, v1)
		}
	}
	return output
}

// searchNeighborsToAdd performs a beam search starting from entryID to find
// candidates for linking a new node at the given level.
// It returns a max-heap of up to efConstruction nearest found nodes.
func (h *HNSW) searchNeighborsToAdd(
	q []float32,
	entryID int32,
	dEntry float32,
	level int,
	visited map[int32]bool,
) *maxHeap {
	// candidates: min-heap, top = nearest unprocessed
	candidates := &minHeap{{dist: dEntry, id: entryID}}
	heap.Init(candidates)

	// results: max-heap, top = farthest kept result
	results := &maxHeap{{dist: dEntry, id: entryID}}
	heap.Init(results)

	visited[entryID] = true

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(nodeDist)

		// Stop when the nearest candidate is farther than the worst result.
		if curr.dist > (*results)[0].dist {
			break
		}

		begin, end := h.neighborRange(int(curr.id), level)
		for j := begin; j < end; j++ {
			nb := h.neighbors[j]
			if nb < 0 {
				break
			}
			if visited[nb] {
				continue
			}
			visited[nb] = true

			d := h.dist(q, nb)
			if results.Len() < h.EfConstruction || d < (*results)[0].dist {
				heap.Push(candidates, nodeDist{dist: d, id: nb})
				heap.Push(results, nodeDist{dist: d, id: nb})
				if results.Len() > h.EfConstruction {
					heap.Pop(results)
				}
			}
		}
	}
	return results
}

// addLink adds a directed edge from src to dest at the given level.
// If the neighbor list is full it runs the shrink heuristic.
func (h *HNSW) addLink(src, dest int32, level int) {
	begin, end := h.neighborRange(int(src), level)

	// Find an empty slot.
	if h.neighbors[end-1] == -1 {
		for i := end - 1; i >= begin; i-- {
			if i == begin || h.neighbors[i-1] != -1 {
				h.neighbors[i] = dest
				return
			}
		}
	}

	// All slots are taken: rebuild the list with the shrink heuristic.
	q := h.vectors[int(src)*h.d : int(src)*h.d+h.d]

	candidates := &maxHeap{{dist: h.symmetricDist(src, dest), id: dest}}
	for i := begin; i < end; i++ {
		nb := h.neighbors[i]
		if nb >= 0 {
			heap.Push(candidates, nodeDist{dist: h.symmetricDist(src, nb), id: nb})
		}
	}
	heap.Init(candidates)

	maxSize := end - begin
	kept := h.shrinkNeighborList(q, candidates, maxSize)

	i := begin
	for _, nd := range kept {
		h.neighbors[i] = nd.id
		i++
	}
	for ; i < end; i++ {
		h.neighbors[i] = -1
	}
}

// addLinksStartingFrom searches for neighbors of ptID at the given level
// (starting from nearest/dNearest) and establishes bidirectional links.
func (h *HNSW) addLinksStartingFrom(
	q []float32,
	ptID int32,
	nearest int32,
	dNearest float32,
	level int,
	visited map[int32]bool,
) {
	linkTargets := h.searchNeighborsToAdd(q, nearest, dNearest, level, visited)

	M := h.nbNeighbors(level)
	kept := h.shrinkNeighborList(q, linkTargets, M)

	for _, nd := range kept {
		h.addLink(ptID, nd.id, level)
		h.addLink(nd.id, ptID, level)
	}
}

// greedyUpdateNearest performs a greedy hill-climb starting from *nearest at
// the given level and updates *nearest and *dNearest to the locally closest
// node found.
func (h *HNSW) greedyUpdateNearest(q []float32, level int, nearest *int32, dNearest *float32) {
	for {
		prev := *nearest
		begin, end := h.neighborRange(int(*nearest), level)
		for j := begin; j < end; j++ {
			v := h.neighbors[j]
			if v < 0 {
				break
			}
			d := h.dist(q, v)
			if d < *dNearest {
				*nearest = v
				*dNearest = d
			}
		}
		if *nearest == prev {
			return
		}
	}
}

// ---------------------------------------------------------------------------
// Public API: Add
// ---------------------------------------------------------------------------

// Add inserts n vectors (rows of x, each of length d) into the index.
// Vectors are stored and indexed in order; the first vector gets ID 0.
func (h *HNSW) Add(n int, x []float32) {
	if n == 0 {
		return
	}
	n0 := len(h.levels) // number of vectors already in the index

	// Assign levels and extend the neighbor table.
	for i := 0; i < n; i++ {
		ptLevel := h.randomLevel()
		h.levels = append(h.levels, ptLevel+1)
		h.offsets = append(h.offsets, h.offsets[len(h.offsets)-1]+h.cumNbNeighbors(ptLevel+1))
	}

	// Extend the neighbors array (fill with -1 = empty).
	newSize := h.offsets[len(h.offsets)-1]
	for len(h.neighbors) < newSize {
		h.neighbors = append(h.neighbors, -1)
	}

	// Append raw vectors.
	h.vectors = append(h.vectors, x[:n*h.d]...)

	// Insert each new point into the graph.
	for i := 0; i < n; i++ {
		ptID := int32(n0 + i)
		ptLevel := h.levels[ptID] - 1
		q := x[i*h.d : i*h.d+h.d]

		if h.entryPoint == -1 {
			h.entryPoint = ptID
			h.maxLevel = ptLevel
			continue
		}

		visited := make(map[int32]bool)
		nearest := h.entryPoint
		dNearest := h.dist(q, nearest)

		// Greedy search on levels above ptLevel to find a good entry point.
		for level := h.maxLevel; level > ptLevel; level-- {
			h.greedyUpdateNearest(q, level, &nearest, &dNearest)
		}

		// Build links from ptLevel down to 0.
		for level := min(ptLevel, h.maxLevel); level >= 0; level-- {
			h.addLinksStartingFrom(q, ptID, nearest, dNearest, level, visited)
			// Update nearest to best found at this level for the next level down.
			begin, end := h.neighborRange(int(ptID), level)
			for j := begin; j < end; j++ {
				nb := h.neighbors[j]
				if nb < 0 {
					break
				}
				d := h.dist(q, nb)
				if d < dNearest {
					nearest = nb
					dNearest = d
				}
			}
		}

		if ptLevel > h.maxLevel {
			h.maxLevel = ptLevel
			h.entryPoint = ptID
		}
	}
}

// ---------------------------------------------------------------------------
// Public API: Search
// ---------------------------------------------------------------------------

// SearchResult holds a single nearest-neighbor result.
type SearchResult struct {
	ID       int32
	Distance float32
}

// Search finds the k approximate nearest neighbors of query q.
// It returns at most k results sorted from nearest to farthest.
func (h *HNSW) Search(q []float32, k int) []SearchResult {
	if h.entryPoint == -1 || k <= 0 {
		return nil
	}

	efSearch := h.EfSearch
	if efSearch < k {
		efSearch = k
	}

	nearest := h.entryPoint
	dNearest := h.dist(q, nearest)

	// Greedy search on upper levels.
	for level := h.maxLevel; level >= 1; level-- {
		h.greedyUpdateNearest(q, level, &nearest, &dNearest)
	}

	// Beam search on level 0.
	results := h.beamSearch(q, nearest, dNearest, 0, efSearch)

	// Extract up to k results, sorted nearest-first.
	out := make([]SearchResult, 0, k)
	// results is a max-heap; drain it into a slice, then reverse.
	tmp := make([]SearchResult, 0, results.Len())
	for results.Len() > 0 {
		nd := heap.Pop(results).(nodeDist)
		tmp = append(tmp, SearchResult{ID: nd.id, Distance: nd.dist})
	}
	// tmp is farthest-first; reverse to get nearest-first.
	for i := len(tmp) - 1; i >= 0 && len(out) < k; i-- {
		out = append(out, tmp[i])
	}
	return out
}

// beamSearch performs the HNSW beam search at a single level and returns
// a max-heap of efSearch nearest nodes found.
func (h *HNSW) beamSearch(
	q []float32,
	entryID int32,
	dEntry float32,
	level, efSearch int,
) *maxHeap {
	visited := make(map[int32]bool)
	visited[entryID] = true

	candidates := &minHeap{{dist: dEntry, id: entryID}}
	heap.Init(candidates)

	results := &maxHeap{{dist: dEntry, id: entryID}}
	heap.Init(results)

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(nodeDist)

		// Pruning: if the nearest unprocessed candidate is already farther
		// than the worst result in the beam, we can stop.
		if curr.dist > (*results)[0].dist {
			break
		}

		begin, end := h.neighborRange(int(curr.id), level)
		for j := begin; j < end; j++ {
			nb := h.neighbors[j]
			if nb < 0 {
				break
			}
			if visited[nb] {
				continue
			}
			visited[nb] = true

			d := h.dist(q, nb)
			if results.Len() < efSearch || d < (*results)[0].dist {
				heap.Push(candidates, nodeDist{dist: d, id: nb})
				heap.Push(results, nodeDist{dist: d, id: nb})
				if results.Len() > efSearch {
					heap.Pop(results)
				}
			}
		}
	}
	return results
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
