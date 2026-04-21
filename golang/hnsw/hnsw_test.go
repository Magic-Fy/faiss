// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

package hnsw

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// ---------------------------------------------------------------------------
// Brute-force helpers used as ground truth
// ---------------------------------------------------------------------------

// bruteForceL2 returns the k nearest neighbors (by squared L2) from vectors
// for the query q. Returns (IDs, distances) sorted nearest-first.
func bruteForceL2(q []float32, vectors []float32, d, k int) ([]int32, []float32) {
	n := len(vectors) / d
	type pair struct {
		id   int32
		dist float32
	}
	all := make([]pair, n)
	for i := 0; i < n; i++ {
		all[i] = pair{int32(i), l2Squared(q, vectors[i*d:(i+1)*d])}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].dist < all[j].dist })
	if k > n {
		k = n
	}
	ids := make([]int32, k)
	dists := make([]float32, k)
	for i := 0; i < k; i++ {
		ids[i] = all[i].id
		dists[i] = all[i].dist
	}
	return ids, dists
}

// recallAt1 computes recall@1: the fraction of queries whose true nearest
// neighbor appears as the first result.
func recallAt1(results []SearchResult, trueNN int32) bool {
	return len(results) > 0 && results[0].ID == trueNN
}

// recallAtK computes the fraction of true top-k neighbors that appear in
// the returned results.
func recallAtK(results []SearchResult, trueIDs []int32) float64 {
	found := make(map[int32]bool, len(results))
	for _, r := range results {
		found[r.ID] = true
	}
	hits := 0
	for _, id := range trueIDs {
		if found[id] {
			hits++
		}
	}
	return float64(hits) / float64(len(trueIDs))
}

// ---------------------------------------------------------------------------
// Basic sanity tests
// ---------------------------------------------------------------------------

func TestAddAndSearchSingle(t *testing.T) {
	idx := New(4, 16, 40, MetricL2)
	x := []float32{1, 2, 3, 4}
	idx.Add(1, x) // Add 1 vector; it is automatically assigned ID 0.
	if idx.Ntotal() != 1 {
		t.Fatalf("expected 1 vector, got %d", idx.Ntotal())
	}
	results := idx.Search(x, 1)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != 0 {
		t.Errorf("expected ID 0, got %d", results[0].ID)
	}
	if results[0].Distance > 1e-6 {
		t.Errorf("expected distance ~0, got %f", results[0].Distance)
	}
}

func TestAddAndSearchTwoVectors(t *testing.T) {
	idx := New(2, 4, 20, MetricL2)
	vectors := []float32{
		0, 0, // id 0
		10, 10, // id 1
	}
	idx.Add(2, vectors)

	q := []float32{0.1, 0.1}
	results := idx.Search(q, 1)
	if len(results) != 1 || results[0].ID != 0 {
		t.Errorf("nearest of (0.1,0.1) should be id 0, got %+v", results)
	}

	q2 := []float32{9.9, 9.9}
	results2 := idx.Search(q2, 1)
	if len(results2) != 1 || results2[0].ID != 1 {
		t.Errorf("nearest of (9.9,9.9) should be id 1, got %+v", results2)
	}
}

func TestSearchEmpty(t *testing.T) {
	idx := New(4, 16, 40, MetricL2)
	results := idx.Search([]float32{1, 2, 3, 4}, 5)
	if len(results) != 0 {
		t.Errorf("expected empty results on empty index, got %d", len(results))
	}
}

func TestSearchKLargerThanN(t *testing.T) {
	idx := New(3, 4, 20, MetricL2)
	x := []float32{1, 2, 3, 4, 5, 6} // 2 vectors
	idx.Add(2, x)
	results := idx.Search([]float32{1, 2, 3}, 10)
	if len(results) > 2 {
		t.Errorf("expected at most 2 results, got %d", len(results))
	}
}

func TestDistancesNonNegative(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	d := 8
	n := 50
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rng.Float32()
	}
	idx := New(d, 8, 40, MetricL2)
	idx.Add(n, vectors)

	for i := 0; i < 10; i++ {
		q := make([]float32, d)
		for j := range q {
			q[j] = rng.Float32()
		}
		results := idx.Search(q, 5)
		for _, r := range results {
			if r.Distance < 0 {
				t.Errorf("negative distance: %f", r.Distance)
			}
		}
	}
}

func TestResultsSortedNearestFirst(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	d := 8
	n := 100
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rng.Float32() * 10
	}
	idx := New(d, 16, 40, MetricL2)
	idx.Add(n, vectors)

	q := make([]float32, d)
	for j := range q {
		q[j] = rng.Float32() * 10
	}
	results := idx.Search(q, 10)
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("results not sorted: results[%d].Distance=%f < results[%d].Distance=%f",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}
}

// ---------------------------------------------------------------------------
// Recall tests (approximate nearest neighbor quality)
// ---------------------------------------------------------------------------

func TestRecall1_L2(t *testing.T) {
	const d = 16
	const n = 500
	const nq = 50
	const k = 1

	rng := rand.New(rand.NewSource(1))
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rng.Float32()
	}

	idx := New(d, 16, 64, MetricL2)
	idx.EfSearch = 64
	idx.Add(n, vectors)

	hits := 0
	for q := 0; q < nq; q++ {
		query := make([]float32, d)
		for j := range query {
			query[j] = rng.Float32()
		}
		results := idx.Search(query, k)
		trueIDs, _ := bruteForceL2(query, vectors, d, k)
		if recallAt1(results, trueIDs[0]) {
			hits++
		}
	}
	recall := float64(hits) / float64(nq)
	t.Logf("recall@1 = %.3f (%d/%d)", recall, hits, nq)
	if recall < 0.90 {
		t.Errorf("recall@1 too low: %.3f", recall)
	}
}

func TestRecallK_L2(t *testing.T) {
	const d = 32
	const n = 1000
	const nq = 50
	const k = 10

	rng := rand.New(rand.NewSource(7))
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rng.Float32()
	}

	idx := New(d, 16, 100, MetricL2)
	idx.EfSearch = 100
	idx.Add(n, vectors)

	var totalRecall float64
	for q := 0; q < nq; q++ {
		query := make([]float32, d)
		for j := range query {
			query[j] = rng.Float32()
		}
		results := idx.Search(query, k)
		trueIDs, _ := bruteForceL2(query, vectors, d, k)
		totalRecall += recallAtK(results, trueIDs)
	}
	avgRecall := totalRecall / float64(nq)
	t.Logf("recall@%d = %.3f", k, avgRecall)
	if avgRecall < 0.85 {
		t.Errorf("recall@%d too low: %.3f", k, avgRecall)
	}
}

func TestRecall1_InnerProduct(t *testing.T) {
	const d = 16
	const n = 500
	const nq = 50
	const k = 1

	rng := rand.New(rand.NewSource(3))

	// Use unit-normalized vectors so inner product ≡ cosine similarity.
	normalize := func(v []float32) {
		var norm float32
		for _, x := range v {
			norm += x * x
		}
		norm = float32(math.Sqrt(float64(norm)))
		for i := range v {
			v[i] /= norm
		}
	}

	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			vectors[i*d+j] = rng.Float32()*2 - 1
		}
		normalize(vectors[i*d : i*d+d])
	}

	idx := New(d, 16, 64, MetricInnerProduct)
	idx.EfSearch = 64
	idx.Add(n, vectors)

	hits := 0
	for q := 0; q < nq; q++ {
		query := make([]float32, d)
		for j := range query {
			query[j] = rng.Float32()*2 - 1
		}
		normalize(query)

		results := idx.Search(query, k)

		// Ground truth: highest inner product.
		type pair struct {
			id   int32
			ip   float32
		}
		all := make([]pair, n)
		for i := 0; i < n; i++ {
			all[i] = pair{int32(i), innerProduct(query, vectors[i*d:(i+1)*d])}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].ip > all[b].ip })

		if len(results) > 0 && results[0].ID == all[0].id {
			hits++
		}
	}
	recall := float64(hits) / float64(nq)
	t.Logf("inner-product recall@1 = %.3f (%d/%d)", recall, hits, nq)
	if recall < 0.85 {
		t.Errorf("inner-product recall@1 too low: %.3f", recall)
	}
}

// ---------------------------------------------------------------------------
// Correctness on a trivial 1-D grid
// ---------------------------------------------------------------------------

func TestExactOnGrid(t *testing.T) {
	// Points at 0, 1, 2, …, 99. Query = 50.2 → nearest = 50.
	n := 100
	vectors := make([]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = float32(i)
	}
	idx := New(1, 8, 40, MetricL2)
	idx.EfSearch = 80
	idx.Add(n, vectors)

	results := idx.Search([]float32{50.2}, 1)
	if len(results) == 0 || results[0].ID != 50 {
		t.Errorf("expected ID 50, got %+v", results)
	}
}

// ---------------------------------------------------------------------------
// Incremental add
// ---------------------------------------------------------------------------

func TestIncrementalAdd(t *testing.T) {
	d := 4
	idx := New(d, 8, 40, MetricL2)

	// Add vectors one at a time.
	rng := rand.New(rand.NewSource(55))
	n := 200
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rng.Float32()
	}
	for i := 0; i < n; i++ {
		idx.Add(1, vectors[i*d:(i+1)*d])
	}
	if idx.Ntotal() != n {
		t.Fatalf("expected %d vectors, got %d", n, idx.Ntotal())
	}

	// Spot-check: query with an exact copy of one vector.
	for _, qID := range []int{0, 99, 199} {
		q := vectors[qID*d : qID*d+d]
		results := idx.Search(q, 1)
		if len(results) == 0 || results[0].ID != int32(qID) {
			t.Errorf("exact query for id %d: got %+v", qID, results)
		}
	}
}
