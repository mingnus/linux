/*
 * Copyright (C) 2021 Red Hat, Inc.
 *
 * This file is released under the GPL.
 */

#include "dm-rtree.h"
#include "dm-space-map.h"

#include <linux/device-mapper.h>

#define DM_MSG_PREFIX "rtree"

/*----------------------------------------------------------------*/

// TODO:
// - We could add a timestamp that is updated every time the block is written (by the validator).
//   This would be helpful for repairing damaged metadata.
// - do we need node_end?

/*----------------------------------------------------------------*/

enum node_flags {
	INTERNAL_NODE = 1,
	LEAF_NODE = 1 << 1
};

struct node_header {
	__le32 csum;
	__le32 flags;
	__le64 blocknr;

	/* this is the end value for the highest range in this subtree */
	__le64 node_end;

	__le32 nr_entries;
	__le32 padding;

} __attribute__((packed, aligned(8)));

// FIXME: if we shrink data_begin to 32bits, then we'll have to aligned(4)
// FIXME: compound the data_begin & len_time into a single le64 to avoid dereferencing twice?
struct disk_mapping {
	__le32 data_begin;
	__le32 len_time;
} __attribute__((packed, aligned(4)));

/*
 * We hold an extra key in these nodes so we know the end value for the highest
 * entry in the children.
 */

#define MD_BLOCK_SIZE 4096
#define INTERNAL_NR_ENTRIES ((MD_BLOCK_SIZE - sizeof(struct node_header)) / (sizeof(__le64) * 2))
#define LEAF_NR_ENTRIES ((MD_BLOCK_SIZE - sizeof(struct node_header))  / (sizeof(__le64) + sizeof(struct disk_mapping)))

// FIXME: make 32bit variants for small keys, and small data_blocks
struct internal_node {
	struct node_header header;
	__le64 keys[INTERNAL_NR_ENTRIES];
	__le64 values[INTERNAL_NR_ENTRIES];
} __attribute__((packed, aligned(8)));

struct leaf_node {
	struct node_header header;
	__le64 keys[LEAF_NR_ENTRIES];
	struct disk_mapping values[LEAF_NR_ENTRIES];
} __attribute__((packed, aligned(8)));

struct del_args {
	struct dm_transaction_manager *tm;
	struct dm_space_map *data_sm;
};

struct insert_args {
	struct dm_transaction_manager *tm;
	struct dm_space_map *data_sm;
	struct dm_mapping *v;
};

struct node_info {
	dm_block_t loc;
	dm_block_t lowest_key;
	unsigned nr_entries;
};

struct insert_result {
	unsigned nr_nodes;
	struct node_info nodes[2];
	unsigned flags;

	// counters
	unsigned action[8];
	unsigned stats[8];
};

enum range_relation {
	DISJOINTED = 0x1,
	ADJACENTED = 0x2,
	OVERLAPPED_ALL = 0x4,
	OVERLAPPED_HEAD = 0x8,
	OVERLAPPED_TAIL = 0x10,
	OVERLAPPED_PARTIAL = 0x20,
	OVERLAPPED_CENTER = 0x40,
};

enum value_compatibility {
	INCOMPATIBLE = 0x100,	// FIXME: do we really need an incompatible flag?
	COMPATIBLE = 0x200,
};

enum leaf_insert_actions {
	NOOP = 0,
	PUSH_BACK,		// LEFT | ADJACENTED | COMPATIBLE,
	PUSH_BACK_REPLACED,
	PUSH_BACK_TRUNCATED,
	PUSH_FRONT,		//  RIGHT | ADJACENTED | COMPATIBLE,
	PUSH_FRONT_REPLACED,
	PUSH_FRONT_TRUNCATED,
	MERGE,			// LEFT | RIGHT | ADJACENTED | COMPATIBLE,
	MERGE_REPLACED,
	TRUNCATE_BACK,		// LEFT | OVERLAPPED_TAIL | INCOMPATIBLE,
	TRUNCATE_FRONT,		// LEFT | OVERLAPPED_HEAD | INCOMPATIBLE,
	REPLACE,		// LEFT | OVERLAP_ALL | INCOMPATIBLE
	INSERT_NEW,		// ADJACENTED-INCOMPATIBLE, DISJOINT-*
	SPLIT,			// LEFT | OVERLAPPED_CENTER | INCOMPATIBLE,
};

struct remove_args {
	struct dm_transaction_manager *tm;
	struct dm_space_map *data_sm;
	dm_block_t thin_b;
	dm_block_t thin_e;
};

struct remove_result {
	unsigned nr_nodes;
	struct node_info nodes[2];
};

struct rebalance_result {
	unsigned nr_nodes;
	struct node_info nodes[3];
};

enum insert_action {
	LEAF_SPLIT = 1,
	LEAF_OVERFLOW,
	LEAF_UNDERFLOW,
	INTERNAL_SPLIT,
	INTERNAL_OVERFLOW,
	INTERNAL_UNDERFLOW,
};

enum rebalance_res {
	NOTHING = 1,
	REDIST2,
	REDIST3,
	MERGED_INTO_ONE,
	MERGED_INTO_TWO,
};

struct node_ops {
	/*
	 * Shifting may cause entries to be merged, so don't assume you know
	 * the nr_entries after a shift.
	 */
	void (*rebalance2)(struct dm_transaction_manager * tm,
			   struct dm_block * left, struct dm_block * right,
			   struct rebalance_result * res);
	void (*rebalance3)(struct dm_transaction_manager * tm,
			   struct dm_block * left, struct dm_block * center,
			   struct dm_block * right,
			   struct rebalance_result * res);
	int (*del)(struct del_args * args, struct dm_block * b);
	int (*insert)(struct insert_args * args, struct dm_block * b,
		      struct insert_result * res);
	int (*remove)(struct remove_args * args, struct dm_block * b);
};

static int get_ops(struct node_header *h, struct node_ops **ops);
static int get_node_free_space(struct dm_transaction_manager *tm, dm_block_t b,
			       unsigned *space);

/*----------------------------------------------------------------*/

// FIXME: generate own random number
#define RTREE_CSUM_XOR 0x8f279d9c

static void node_prep(struct dm_block_validator *v,
		      struct dm_block *b, size_t block_size)
{
	struct node_header *h = dm_block_data(b);

	h->blocknr = cpu_to_le64(dm_block_location(b));
	h->csum = cpu_to_le32(dm_bm_checksum(&h->flags,
					     block_size - sizeof(__le32),
					     RTREE_CSUM_XOR));
}

static int node_check(struct dm_block_validator *v,
		      struct dm_block *b, size_t block_size)
{
	struct node_header *h = dm_block_data(b);
	uint32_t csum;

	if (dm_block_location(b) != le64_to_cpu(h->blocknr)) {
		DMERR_LIMIT("node_check failed: blocknr %llu != wanted %llu",
			    le64_to_cpu(h->blocknr), dm_block_location(b));
		return -ENOTBLK;
	}

	csum =
	    dm_bm_checksum(&h->flags, block_size - sizeof(__le32),
			   RTREE_CSUM_XOR);
	if (csum != le32_to_cpu(h->csum)) {
		DMERR_LIMIT("node_check failed: csum %u != wanted %u", csum,
			    le32_to_cpu(h->csum));
		return -EILSEQ;
	}

	return 0;
}

struct dm_block_validator validator = {
	.name = "rtree_node",
	.prepare_for_write = node_prep,
	.check = node_check
};

/*----------------------------------------------------------------*/

#define TIME_BITS 20
#define LEN_BITS 12

// (1 << LEN_BITS) - 1
#define MAPPINGS_MAX_LEN 4095

static void unpack_len_time(uint32_t len_time, uint32_t *len, uint32_t *time)
{
	*len = len_time >> TIME_BITS;
	*time = len_time & ((1 << TIME_BITS) - 1);
}

static uint32_t pack_len_time(uint32_t len, uint32_t time)
{
	return len << TIME_BITS | time;
}

static int get_mapping(struct leaf_node *n, int i, struct dm_mapping *out)
{
	struct disk_mapping *value_le = n->values + i;
	out->thin_begin = le64_to_cpu(n->keys[i]);
	out->data_begin = le32_to_cpu(value_le->data_begin);
	unpack_len_time(le32_to_cpu(value_le->len_time), &out->len, &out->time);
	return 0;
}

// FIXME: accepts disk_mapping rather than leaf_node
static void set_len(struct leaf_node *n, int i, uint32_t len)
{
	struct disk_mapping *value_le = n->values + i;
	uint32_t len_time = le32_to_cpu(value_le->len_time);
	len_time &= (1 << 20) - 1;
	len_time |= (len << 20);
	value_le->len_time = cpu_to_le32(len_time);
}

/*----------------------------------------------------------------*/

static void init_node_info(struct node_info *info, dm_block_t loc,
			   dm_block_t lowest_key, unsigned nr_entries)
{
	info->loc = loc;
	info->lowest_key = lowest_key;
	info->nr_entries = nr_entries;
}

static int bsearch(uint64_t *keys, unsigned count, uint64_t key, bool want_hi)
{
	int lo = -1, hi = count;

	while (hi - lo > 1) {
		int mid = lo + ((hi - lo) / 2);
		uint64_t mid_key = le64_to_cpu(keys[mid]);

		if (mid_key == key)
			return mid;

		if (mid_key < key)
			lo = mid;
		else
			hi = mid;
	}

	return want_hi ? hi : lo;
}

static inline int lower_bound(uint64_t *keys, unsigned count, uint64_t key)
{
	return bsearch(keys, count, key, false);
}

static void array_insert(void *base, size_t elt_size, unsigned nr_elts,
			 unsigned index, void *elt)
{
	if (index < nr_elts)
		memmove(base + (elt_size * (index + 1)),
			base + (elt_size * index),
			(nr_elts - index) * elt_size);

	memcpy(base + (elt_size * index), elt, elt_size);
}

static void array_erase(void *base, size_t elt_size, unsigned nr_elts,
			unsigned index)
{
	memmove(base + (elt_size * index),
		base + (elt_size * (index + 1)),
		(nr_elts - index - 1) * elt_size);
}

/*----------------------------------------------------------------*/

static int del_(struct del_args *args, dm_block_t loc)
{
	int r;
	struct node_ops *ops;
	struct dm_block *b;
	int shared;

	r = dm_tm_block_is_shared(args->tm, loc, &shared);
	if (r)
		return r;

	if (shared) {
		/* just decrement the ref count for this block */
		dm_tm_dec(args->tm, loc);
		return 0;
	}

	r = dm_tm_read_lock(args->tm, loc, &validator, &b);
	if (r)
		return r;

	r = get_ops(dm_block_data(b), &ops);
	if (!r)
		r = ops->del(args, b);

	dm_tm_unlock(args->tm, b);

	if (!r)
		dm_tm_dec(args->tm, loc);	// FIXME: return errors form dm_tm_dec()?

	return r;
}

/*----------------------------------------------------------------*/

/**
 * inc_children - Increment the reference count of all children of an rtree
 * node.
 *
 * @tm: The transaction manager to use.
 * @data_sm: The data space map to use.
 * @b: The block containing the node to increment the children of.
 *
 * Returns:
 *  - 0 on success
 *  - -EINVAL if the node is not an internal or leaf node
 */
static int inc_children(struct dm_transaction_manager *tm,
			struct dm_space_map *data_sm, struct dm_block *b)
{
	struct node_header *h = dm_block_data(b);
	uint32_t flags = le32_to_cpu(h->flags);
	uint32_t nr_entries = le32_to_cpu(h->nr_entries);

	if (flags == INTERNAL_NODE) {
		struct internal_node *n = (struct internal_node *)h;
		dm_tm_with_runs(tm, n->values, nr_entries, dm_tm_inc_range);

	} else if (flags == LEAF_NODE) {
		struct leaf_node *n = (struct leaf_node *)h;
		int i;
		for (i = 0; i < nr_entries;
		     i++) { // FIXME: loop through the value ptr
			struct dm_mapping m;
			dm_block_t data_end;
			get_mapping(n, i, &m);
			data_end = m.data_begin + m.len;
			dm_sm_inc_blocks(data_sm, m.data_begin, data_end);
		}
	} else {
		return -EINVAL;
	}

	return 0;
}

/**
 * shadow_node - Shadow a node and increment any children.
 *
 * @tm: The transaction manager to use.
 * @data_sm: The data space map to use.
 * @loc: The location of the block to shadow.
 * @result: Pointer to a variable that will be set to the shadow block.
 */
static int shadow_node(struct dm_transaction_manager *tm,
			   struct dm_space_map *data_sm, dm_block_t loc,
			   struct dm_block **result)
{
	int inc, r;

	r = dm_tm_shadow_block(tm, loc, &validator, result, &inc);
	if (r)
		return r;

	if (inc) {
		r = inc_children(tm, data_sm, *result);
		if (r) {
			dm_tm_unlock(tm, *result);
			dm_tm_dec(tm, dm_block_location(*result));
			return r;
		}
	}

	return 0;
}

/*----------------------------------------------------------------*/

static void internal_erase(struct internal_node *n, unsigned index)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	if (index < nr_entries - 1) {
		array_erase(n->keys, sizeof(n->keys[0]), nr_entries, index);
		array_erase(n->values, sizeof(n->values[0]), nr_entries, index);
	}
	n->header.nr_entries = cpu_to_le32(nr_entries - 1);
}

static void internal_move(struct internal_node *n, int shift)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);

	if (shift < 0) {
		shift = -shift;
		BUG_ON(shift > nr_entries);
		memmove(n->keys, n->keys + shift,
			(nr_entries - shift) * sizeof(n->keys[0]));
		memmove(n->values, n->values + shift,
			(nr_entries - shift) * sizeof(n->values[0]));
	} else {
		BUG_ON(nr_entries + shift > INTERNAL_NR_ENTRIES);
		memmove(n->keys + shift, n->keys,
			nr_entries * sizeof(n->keys[0]));
		memmove(n->values + shift, n->values,
			nr_entries * sizeof(n->values[0]));
	}
}

static void internal_copy(struct internal_node *left,
			  struct internal_node *right, int shift)
{
	uint32_t nr_left = le32_to_cpu(left->header.nr_entries);

	if (shift < 0) {
		shift = -shift;
		BUG_ON(nr_left + shift > INTERNAL_NR_ENTRIES);
		memcpy(left->keys + nr_left, right->keys,
		       shift * sizeof(left->keys[0]));
		memcpy(left->values + nr_left, right->values,
		       shift * sizeof(left->values[0]));
	} else {
		BUG_ON(shift > INTERNAL_NR_ENTRIES);
		memcpy(right->keys, left->keys + (nr_left - shift),
		       shift * sizeof(left->keys[0]));
		memcpy(right->values, left->values + (nr_left - shift),
		       shift * sizeof(left->values[0]));
	}
}

static void internal_shift(struct dm_block *left, struct dm_block *right,
			   int count)
{
	struct internal_node *l = dm_block_data(left);
	struct internal_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	BUG_ON(nr_left - count > INTERNAL_NR_ENTRIES);
	BUG_ON(nr_right + count > INTERNAL_NR_ENTRIES);

	if (count > 0) {
		internal_move(r, count);
		internal_copy(l, r, count);
	} else {
		internal_copy(l, r, count);
		internal_move(r, count);
	}

	l->header.nr_entries = cpu_to_le32(nr_left - count);
	r->header.nr_entries = cpu_to_le32(nr_right + count);
}

static void redist2_internal(struct dm_block *left_, struct dm_block *right_);
static void internal_rebalance2(struct dm_transaction_manager *tm,
				struct dm_block *left, struct dm_block *right,
				struct rebalance_result *res)
{
	struct internal_node *l = dm_block_data(left);
	struct internal_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	if (nr_left + nr_right <= INTERNAL_NR_ENTRIES) {
		/* merge the two nodes */
		internal_shift(left, right, -le32_to_cpu(r->header.nr_entries));
		dm_tm_dec(tm, dm_block_location(right));
		res->nr_nodes = 1;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));

	} else {
		redist2_internal(left, right);
		res->nr_nodes = 2;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));
	}
}

static void internal_delete_center_node(struct dm_transaction_manager *tm,
					struct dm_block *left,
					struct dm_block *center,
					struct dm_block *right)
{
	struct internal_node *l = dm_block_data(left);
	struct internal_node *c = dm_block_data(center);
	struct internal_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_center = le32_to_cpu(c->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	unsigned shift =
	    min((uint32_t) INTERNAL_NR_ENTRIES - nr_left, nr_center);

	BUG_ON(nr_left + shift > INTERNAL_NR_ENTRIES);
	internal_shift(left, center, -shift);

	if (shift != nr_center) {
		shift = nr_center - shift;
		BUG_ON((nr_right + shift) > INTERNAL_NR_ENTRIES);
		internal_shift(center, right, shift);
	}

	dm_tm_dec(tm, dm_block_location(center));
	redist2_internal(left, right);
}

static void redist3_internal(struct dm_block *left_, struct dm_block *center_,
			     struct dm_block *right_);
static void internal_rebalance3(struct dm_transaction_manager *tm,
				struct dm_block *left, struct dm_block *center,
				struct dm_block *right,
				struct rebalance_result *res)
{
	struct internal_node *l = dm_block_data(left);
	struct internal_node *c = dm_block_data(center);
	struct internal_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_center = le32_to_cpu(c->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	unsigned threshold = (INTERNAL_NR_ENTRIES - 8) * 2;

	if (nr_left + nr_center + nr_right <= threshold) {
		internal_delete_center_node(tm, left, center, right);
		res->nr_nodes = 2;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));

	} else {
		redist3_internal(left, center, right);
		res->nr_nodes = 3;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(center),
			       le64_to_cpu(c->keys[0]),
			       le32_to_cpu(c->header.nr_entries));
		init_node_info(&res->nodes[2], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));
	}
}

static int internal_del(struct del_args *args, struct dm_block *b)
{
	struct internal_node *n;
	struct dm_block_manager *bm;
	uint32_t i, nr_entries;
	int r;

	n = dm_block_data(b);
	nr_entries = le32_to_cpu(n->header.nr_entries);
	bm = dm_tm_get_bm(args->tm);

	/* prefetch children */
	for (i = 0; i < nr_entries; i++)
		dm_bm_prefetch(bm, le64_to_cpu(n->values[i]));

	/* recurse into children */
	for (i = 0; i < nr_entries; i++) {
		dm_block_t child_b = le64_to_cpu(n->values[i]);
		r = del_(args, child_b);
		if (r)
			return r;
	}

	return 0;
}

static void insert_into_internal(struct internal_node *node, unsigned index,
				 uint64_t key, dm_block_t value)
{
	uint32_t nr_entries = le32_to_cpu(node->header.nr_entries);
	__le64 key_le = cpu_to_le64(key);
	__le64 value_le = cpu_to_le64(value);

	array_insert(node->keys, sizeof(node->keys[0]), nr_entries, index,
		     &key_le);
	array_insert(node->values, sizeof(node->values[0]), nr_entries, index,
		     &value_le);
	node->header.nr_entries = cpu_to_le32(nr_entries + 1);
}

static void redist2_internal(struct dm_block *left_, struct dm_block *right_)
{
	struct internal_node *left = dm_block_data(left_);
	struct internal_node *right = dm_block_data(right_);
	unsigned nr_left = le32_to_cpu(left->header.nr_entries);
	unsigned nr_right = le32_to_cpu(right->header.nr_entries);
	unsigned total = nr_left + nr_right;
	unsigned target_left = total / 2;

	// FIXME: these are the same
	if (nr_left < target_left) {
		int delta = (int)nr_left - (int)target_left;
		internal_shift(left_, right_, delta);

	} else if (nr_left > target_left) {
		int delta = nr_left - target_left;
		internal_shift(left_, right_, delta);
	}
}

static void redist3_internal(struct dm_block *left_, struct dm_block *center_,
			     struct dm_block *right_)
{
	struct internal_node *left = dm_block_data(left_);
	struct internal_node *center = dm_block_data(center_);
	struct internal_node *right = dm_block_data(right_);

	unsigned nr_left = le32_to_cpu(left->header.nr_entries);
	unsigned nr_center = le32_to_cpu(center->header.nr_entries);
	unsigned nr_right = le32_to_cpu(right->header.nr_entries);

	unsigned total = nr_left + nr_right;
	unsigned target_right = total / 3;
	unsigned remainder = (target_right * 3) != total;
	unsigned target_left = target_right + remainder;

	// FIXME: these are the same
	if (nr_left < nr_right) {
		int delta = (int)nr_left - (int)target_left;

		if (delta < 0 && nr_center < -delta) {
			internal_shift(left_, center_, -nr_center);
			delta += nr_center;
			internal_shift(left_, right_, delta);
			nr_right += delta;
		} else {
			internal_shift(left_, center_, delta);
		}

		internal_shift(center_, right_, target_right - nr_right);
	} else if (nr_left > target_left) {
		int delta = (int)target_right - (int)nr_right;

		if (delta > 0 && nr_center < delta) {
			internal_shift(center_, right_, nr_center);
			delta -= nr_center;
			internal_shift(left_, right_, delta);
			nr_left -= delta;
		} else {
			internal_shift(center_, right_, delta);
		}
	}
}


// FIXME: rename
static int insert_aux(struct insert_args *args, dm_block_t loc,
		      struct insert_result *res)
{
	int r;
	struct dm_block *b;
	struct node_ops *ops;

	r = shadow_node(args->tm, args->data_sm, loc, &b);
	if (r)
		return r;

	r = get_ops(dm_block_data(b), &ops);
	if (r) {
		dm_tm_unlock(args->tm, b);
		return r;
	}

	r = ops->insert(args, b, res);
	dm_tm_unlock(args->tm, b);
	return r;
}

static int rebalance3(struct insert_args *args, struct internal_node *n,
		      unsigned index, int *result)
{
	int r;
	struct rebalance_result res;
	struct dm_block *left, *center, *right;
	struct node_ops *ops;

	r = shadow_node(args->tm, args->data_sm, n->values[index], &left);
	if (r)
		return r;

	r = shadow_node(args->tm, args->data_sm, n->values[index + 1], &center);
	if (r) {
		dm_tm_unlock(args->tm, left);
		return r;
	}

	r = shadow_node(args->tm, args->data_sm, n->values[index + 2], &right);
	if (r) {
		dm_tm_unlock(args->tm, left);
		dm_tm_unlock(args->tm, center);
		return r;
	}

	r = get_ops(dm_block_data(left), &ops);
	if (r)
		goto out;

	ops->rebalance3(args->tm, left, center, right, &res);

	if (res.nr_nodes == 2) {
		n->keys[index] = res.nodes[0].lowest_key;
		n->values[index] = res.nodes[0].loc;

		internal_erase(n, index + 1);

		n->keys[index + 1] = res.nodes[1].lowest_key;
		n->values[index + 1] = res.nodes[1].loc;

		*result = MERGED_INTO_TWO;
	} else {
		n->keys[index] = res.nodes[0].lowest_key;
		n->values[index] = res.nodes[0].loc;
		n->keys[index + 1] = res.nodes[1].lowest_key;
		n->values[index + 1] = res.nodes[1].loc;
		n->keys[index + 2] = res.nodes[2].lowest_key;
		n->values[index + 2] = res.nodes[2].loc;

		*result = MERGED_INTO_ONE;
	}

 out:
	dm_tm_unlock(args->tm, left);
	dm_tm_unlock(args->tm, center);
	dm_tm_unlock(args->tm, right);
	return r;
}

static int rebalance2(struct insert_args *args, struct internal_node *n,
		      unsigned index, int *result)
{
	int r;
	struct rebalance_result res;
	struct dm_block *left, *right;
	struct node_ops *ops;

	r = shadow_node(args->tm, args->data_sm, n->values[index], &left);
	if (r)
		return r;

	r = shadow_node(args->tm, args->data_sm, n->values[index + 1], &right);
	if (r) {
		dm_tm_unlock(args->tm, left);
		return r;
	}

	r = get_ops(dm_block_data(left), &ops);
	if (r)
		goto out;

	ops->rebalance2(args->tm, left, right, &res);

	if (res.nr_nodes == 1) {
		if (le32_to_cpu(n->header.nr_entries == 2)) {
			// FIXME: avoid this copy
			memcpy(n, dm_block_data(left), 4096);
			dm_tm_dec(args->tm, dm_block_location(left));
		} else {
			n->keys[index] = res.nodes[0].lowest_key;
			n->values[index] = res.nodes[0].loc;
			internal_erase(n, index + 1);
		}

		*result = MERGED_INTO_ONE;
	} else {
		n->keys[index] = res.nodes[0].lowest_key;
		n->values[index] = res.nodes[0].loc;
		n->keys[index + 1] = res.nodes[1].lowest_key;
		n->values[index + 1] = res.nodes[1].loc;

		*result = REDIST2;
	}

 out:
	dm_tm_unlock(args->tm, left);
	dm_tm_unlock(args->tm, right);
	return r;
}

static int refill(struct insert_args *args, struct internal_node *n,
		  unsigned index, int *res)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	BUG_ON(nr_entries < 2);

	if (index == 0)
		rebalance2(args, n, 0, res);
	else if (index == nr_entries - 1)
		rebalance2(args, n, nr_entries - 2, res);
	else
		rebalance3(args, n, index - 1, res);

	return 0;
}

static int rebalance(struct insert_args *args, struct internal_node *n,
		     unsigned index, int *res)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	unsigned free_space;
	int r;

	BUG_ON(nr_entries < 2);

	if (index > 0) {
		r = get_node_free_space(args->tm, n->values[index - 1],
					&free_space);
		if (r)
			return r;

		if (free_space > 8) {
			rebalance2(args, n, index - 1, res);
			return 0;
		}
	}

	if (index < nr_entries - 1) {
		r = get_node_free_space(args->tm, n->values[index + 1],
					&free_space);
		if (r)
			return r;

		if (free_space > 8) {
			rebalance2(args, n, index, res);
		} else {
			*res = NOTHING;
		}
	} else {
		*res = NOTHING;
	}

	return 0;
}

static int internal_insert(struct insert_args *args, struct dm_block *b,
			   struct insert_result *res)
{
	int r, i;
	dm_block_t child_b;
	struct internal_node *n = dm_block_data(b);
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	int rebalance_outcome = 0;

	i = lower_bound(n->keys, nr_entries, args->v->thin_begin);
	if (i < 0)
		i = 0;

	child_b = le64_to_cpu(n->values[i]);
	r = insert_aux(args, child_b, res);

	if (res->nr_nodes == 1) {
		nr_entries = res->nodes[0].nr_entries;

		n->keys[i] = cpu_to_le64(res->nodes[0].lowest_key);
		n->values[i] = cpu_to_le64(res->nodes[0].loc);

		// FIXME: this should depend on whether it's a leaf or internal
		if (res->flags == INTERNAL_NODE) {
			if (nr_entries == INTERNAL_NR_ENTRIES) {
				rebalance(args, n, i, &rebalance_outcome);
				res->action[INTERNAL_OVERFLOW]++;
			} else if (nr_entries < 100) {
				//rebalance(args, n, i);
				refill(args, n, i, &rebalance_outcome);
				res->action[INTERNAL_UNDERFLOW]++;
			}
		} else if (nr_entries == LEAF_NR_ENTRIES) {
			rebalance(args, n, i, &rebalance_outcome);
			res->action[LEAF_OVERFLOW]++;
		} else if (nr_entries < 80) {
			//rebalance(args, n, i);
			refill(args, n, i, &rebalance_outcome);
			res->action[LEAF_UNDERFLOW]++;
		}

		res->nr_nodes = 1;
		init_node_info(&res->nodes[0], dm_block_location(b),
			       le64_to_cpu(n->keys[0]),
			       le32_to_cpu(n->header.nr_entries));

		if (rebalance_outcome > 0)
			res->stats[rebalance_outcome]++;
	} else {
		n->keys[i] = cpu_to_le64(res->nodes[0].lowest_key);
		n->values[i] = cpu_to_le64(res->nodes[0].loc);

		if (nr_entries < INTERNAL_NR_ENTRIES) {
			insert_into_internal(n, i + 1, res->nodes[1].lowest_key,
					     res->nodes[1].loc);
			res->nr_nodes = 1;
			init_node_info(&res->nodes[0], dm_block_location(b),
				       le64_to_cpu(n->keys[0]),
				       le32_to_cpu(n->header.nr_entries));

		} else {
			/* split the node */
			struct dm_block *sib;
			struct internal_node *sib_n, *n2;

			r = dm_tm_new_block(args->tm, &validator, &sib);
			if (r < 0)
				return r;

			sib_n = dm_block_data(sib);

			sib_n->header.flags = n->header.flags;
			sib_n->header.nr_entries = cpu_to_le32(0);

			// TODO: try rebalancing with the sibling whilst insertion
			// (i.e., split_two_into_three)
			redist2_internal(b, sib);

			if (args->v->thin_begin < le64_to_cpu(sib_n->keys[0]))
				n2 = n;
			else {
				n2 = sib_n;
				i -= le32_to_cpu(n->header.nr_entries);
			}
			insert_into_internal(n2, i + 1,
					     res->nodes[1].lowest_key,
					     res->nodes[1].loc);

			res->nr_nodes = 2;
			init_node_info(&res->nodes[0], dm_block_location(b),
				       le64_to_cpu(n->keys[0]),
				       le32_to_cpu(n->header.nr_entries));
			init_node_info(&res->nodes[1], dm_block_location(sib),
				       le64_to_cpu(sib_n->keys[0]),
				       le32_to_cpu(sib_n->header.nr_entries));

			dm_tm_unlock(args->tm, sib);
			res->action[INTERNAL_SPLIT]++;
		}
	}

	res->flags = INTERNAL_NODE;

	return 0;
}

static int internal_remove(struct remove_args *args, struct dm_block *b)
{
	return -EINVAL;
}

static struct node_ops internal_ops = {
	.rebalance2 = internal_rebalance2,
	.rebalance3 = internal_rebalance3,
	.del = internal_del,
	.insert = internal_insert,
	.remove = internal_remove
};

/*----------------------------------------------------------------*/

static bool adjacent_mapping(struct dm_mapping *left, struct dm_mapping *right)
{
	uint64_t thin_delta = right->thin_begin - left->thin_begin;

	return (thin_delta == left->len) &&
	    (left->data_begin + (uint64_t) left->len == right->data_begin) &&
	    (left->time == right->time);
}

static int test_relation(struct dm_mapping *left, struct dm_mapping *right)
{
	uint64_t thin_delta = right->thin_begin - left->thin_begin;
	int flags = 0;

	if ((left->data_begin + thin_delta == right->data_begin) &&
	    (left->time == right->time))
		flags |= COMPATIBLE;
	else
		flags |= INCOMPATIBLE;

	if (thin_delta == left->len)
		flags |= ADJACENTED;
	else if (thin_delta > left->len)
		flags |= DISJOINTED;
	else {
		uint64_t left_end = left->thin_begin + (uint64_t) left->len;
		uint64_t right_end = right->thin_begin + (uint64_t) right->len;
		if (thin_delta == 0 && left_end == right_end)
			flags |= OVERLAPPED_ALL;
		else if (thin_delta == 0)
			flags |= OVERLAPPED_HEAD;
		else if (left_end == right_end)
			flags |= OVERLAPPED_TAIL;
		else if (left_end < right_end)
			flags |= OVERLAPPED_PARTIAL;
		else
			flags |= OVERLAPPED_CENTER;
	}

	return flags;
}

static void leaf_move(struct leaf_node *n, int shift)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);

	if (shift == 0) {

	} else if (shift < 0) {
		shift = -shift;
		BUG_ON(shift > nr_entries);
		memmove(n->keys, n->keys + shift,
			(nr_entries - shift) * sizeof(n->keys[0]));
		memmove(n->values, n->values + shift,
			(nr_entries - shift) * sizeof(n->values[0]));
	} else {
		BUG_ON(nr_entries + shift > LEAF_NR_ENTRIES);
		memmove(n->keys + shift, n->keys,
			nr_entries * sizeof(n->keys[0]));
		memmove(n->values + shift, n->values,
			nr_entries * sizeof(n->values[0]));
	}
}

static void leaf_copy(struct leaf_node *left, struct leaf_node *right,
		      int shift)
{
	uint32_t nr_left = le32_to_cpu(left->header.nr_entries);

	if (shift < 0) {
		shift = -shift;
		BUG_ON(nr_left + shift > LEAF_NR_ENTRIES);
		memcpy(left->keys + nr_left, right->keys,
		       shift * sizeof(left->keys[0]));
		memcpy(left->values + nr_left, right->values,
		       shift * sizeof(left->values[0]));
	} else {
		BUG_ON(shift > LEAF_NR_ENTRIES);
		memcpy(right->keys, left->keys + (nr_left - shift),
		       shift * sizeof(left->keys[0]));
		memcpy(right->values, left->values + (nr_left - shift),
		       shift * sizeof(left->values[0]));
	}
}

static void leaf_shift_(struct leaf_node *l, struct leaf_node *r, int count)
{
	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	BUG_ON(nr_left - count > LEAF_NR_ENTRIES);
	BUG_ON(nr_right + count > LEAF_NR_ENTRIES);

	if (count > 0) {
		leaf_move(r, count);
		leaf_copy(l, r, count);
	} else {
		leaf_copy(l, r, count);
		leaf_move(r, count);
	}

	l->header.nr_entries = cpu_to_le32(nr_left - count);
	r->header.nr_entries = cpu_to_le32(nr_right + count);
}

static void leaf_shift(struct dm_block *left, struct dm_block *right, int count)
{
	struct leaf_node *l = dm_block_data(left);
	struct leaf_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	if (!count)
		return;

	if (nr_left && nr_right) {
		/*
		 * Check to see if we can merge the last entry from the left
		 * node with the first entry from the right node.
		 */
		struct dm_mapping ll;	// last left
		struct dm_mapping fr;	// first right
		get_mapping(l, nr_left - 1, &ll);
		get_mapping(r, 0, &fr);

		if (adjacent_mapping(&ll, &fr)
		    && ll.len + fr.len <= MAPPINGS_MAX_LEN) {
			l->header.nr_entries = cpu_to_le32(nr_left - 1);
			r->keys[0] = l->keys[nr_left - 1];
			r->values[0].data_begin =
			    l->values[nr_left - 1].data_begin;
			set_len(r, 0, ll.len + fr.len);

			// Adjust the count, since there's one fewer entries in left now
			if (count > 0)
				count -= 1;
		}
	}

	leaf_shift_(l, r, count);
}

static void redist2_leaf(struct dm_block *left_, struct dm_block *right_);
static void leaf_rebalance2(struct dm_transaction_manager *tm,
			    struct dm_block *left, struct dm_block *right,
			    struct rebalance_result *res)
{
	struct leaf_node *l = dm_block_data(left);
	struct leaf_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	if (nr_left + nr_right <= LEAF_NR_ENTRIES - 8) {
		/* merge the two nodes */
		leaf_shift(left, right, -nr_right);
		dm_tm_dec(tm, dm_block_location(right));
		res->nr_nodes = 1;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));

	} else {
		redist2_leaf(left, right);
		res->nr_nodes = 2;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));
	}
}

static void leaf_delete_center_node(struct dm_transaction_manager *tm,
				    struct dm_block *left,
				    struct dm_block *center,
				    struct dm_block *right)
{
	struct leaf_node *l = dm_block_data(left);
	struct leaf_node *c = dm_block_data(center);
	struct leaf_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_center = le32_to_cpu(c->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	unsigned shift =
	    min((uint32_t) LEAF_NR_ENTRIES - nr_left - 8, nr_center);

	BUG_ON(nr_left + shift > LEAF_NR_ENTRIES);
	leaf_shift(left, center, -shift);

	if (shift != nr_center) {
		shift = nr_center - shift;
		BUG_ON((nr_right + shift) > LEAF_NR_ENTRIES);
		leaf_shift(center, right, shift);
	}

	dm_tm_dec(tm, dm_block_location(center));
	redist2_leaf(left, right);
}

static void redist3_leaf(struct dm_block *left_, struct dm_block *center_,
			 struct dm_block *right_);
static void leaf_rebalance3(struct dm_transaction_manager *tm,
			    struct dm_block *left, struct dm_block *center,
			    struct dm_block *right,
			    struct rebalance_result *res)
{
	struct leaf_node *l = dm_block_data(left);
	struct leaf_node *c = dm_block_data(center);
	struct leaf_node *r = dm_block_data(right);

	uint32_t nr_left = le32_to_cpu(l->header.nr_entries);
	uint32_t nr_center = le32_to_cpu(c->header.nr_entries);
	uint32_t nr_right = le32_to_cpu(r->header.nr_entries);

	unsigned threshold = (LEAF_NR_ENTRIES - 8) * 2;	// TODO: try different threshold

	if (nr_left + nr_center + nr_right <= threshold) {
		leaf_delete_center_node(tm, left, center, right);
		res->nr_nodes = 2;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));

	} else {
		redist3_leaf(left, center, right);
		res->nr_nodes = 3;
		init_node_info(&res->nodes[0], dm_block_location(left),
			       le64_to_cpu(l->keys[0]),
			       le32_to_cpu(l->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(center),
			       le64_to_cpu(c->keys[0]),
			       le32_to_cpu(c->header.nr_entries));
		init_node_info(&res->nodes[2], dm_block_location(right),
			       le64_to_cpu(r->keys[0]),
			       le32_to_cpu(r->header.nr_entries));
	}
}

static int leaf_del(struct del_args *args, struct dm_block *b)
{
	struct leaf_node *n;
	uint32_t i, nr_entries;
	int r;

	n = dm_block_data(b);
	nr_entries = le32_to_cpu(n->header.nr_entries);

	/* release the data blocks */
	for (i = 0; i < nr_entries; i++) {
		struct disk_mapping *value_le = n->values + i;
		uint32_t len;
		uint32_t time;
		dm_block_t data_begin;
		dm_block_t data_end;

		unpack_len_time(value_le->len_time, &len, &time);
		data_begin = le64_to_cpu(value_le->data_begin);
		data_end = data_begin + len;
		r = dm_sm_dec_blocks(args->data_sm, data_begin, data_end);
		if (r)
			return r;
	}

	return 0;
}

// FIXME: make copy/move_entries generic, then we need only one redist2
// fn.
static void redist2_leaf(struct dm_block *left_, struct dm_block *right_)
{
	struct leaf_node *left = dm_block_data(left_);
	struct leaf_node *right = dm_block_data(right_);
	unsigned nr_left = le32_to_cpu(left->header.nr_entries);
	unsigned nr_right = le32_to_cpu(right->header.nr_entries);
	unsigned total = nr_left + nr_right;
	unsigned target_left = total / 2;

	// FIXME: these are the same
	if (nr_left < target_left) {
		int delta = (int)nr_left - (int)target_left;
		leaf_shift(left_, right_, delta);

	} else if (nr_left > target_left) {
		int delta = nr_left - target_left;
		leaf_shift(left_, right_, delta);
	}
}

static void redist3_leaf(struct dm_block *left_, struct dm_block *center_,
			 struct dm_block *right_)
{
	struct leaf_node *left = dm_block_data(left_);
	struct leaf_node *center = dm_block_data(center_);
	struct leaf_node *right = dm_block_data(right_);

	unsigned nr_left = le32_to_cpu(left->header.nr_entries);
	unsigned nr_center = le32_to_cpu(center->header.nr_entries);
	unsigned nr_right = le32_to_cpu(right->header.nr_entries);

	unsigned total = nr_left + nr_center + nr_right;
	unsigned target_right = total / 3;
	unsigned remainder = (target_right * 3) != total;
	unsigned target_left = target_right + remainder;

	// FIXME: these are the same
	if (nr_left < nr_right) {
		int delta = (int)nr_left - (int)target_left;

		if (delta < 0 && nr_center < -delta) {
			leaf_shift(left_, center_, -nr_center);
			delta += nr_center;
			leaf_shift(left_, right_, delta);
			nr_right += delta;
		} else {
			leaf_shift(left_, center_, delta);
		}

		leaf_shift(center_, right_, target_right - nr_right);
	} else if (nr_left > target_left) {
		int delta = (int)target_right - (int)nr_right;

		if (delta > 0 && nr_center < delta) {
			leaf_shift(center_, right_, nr_center);
			delta -= nr_center;
			leaf_shift(left_, right_, delta);
			nr_left -= delta;
		} else {
			leaf_shift(center_, right_, delta);
		}
	}
}

static void leaf_insert_(struct leaf_node *n, struct dm_mapping *v,
			 unsigned index)
{
	struct disk_mapping value_le;
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	__le64 key_le = cpu_to_le64(v->thin_begin);

	value_le.data_begin = cpu_to_le64(v->data_begin);
	value_le.len_time = cpu_to_le32(pack_len_time(v->len, v->time));

	array_insert(n->keys, sizeof(n->keys[0]), nr_entries, index, &key_le);
	array_insert(n->values, sizeof(n->values[0]), nr_entries, index,
		     &value_le);
	n->header.nr_entries = cpu_to_le32(nr_entries + 1);
}

static int insert_into_leaf(struct insert_args *args, struct dm_block *b,
			    unsigned index, struct insert_result *res)
{
	int r;
	struct leaf_node *n = dm_block_data(b);
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);

	if (nr_entries == LEAF_NR_ENTRIES) {
		/* split the node */
		struct dm_block *sib;
		struct leaf_node *sib_n, *n2;

		r = dm_tm_new_block(args->tm, &validator, &sib);
		if (r < 0)
			return r;

		sib_n = dm_block_data(sib);

		sib_n->header.flags = n->header.flags;
		sib_n->header.nr_entries = cpu_to_le32(0);

		// TODO: try rebalancing with the sibling whilst insertion
		// (i.e., split_two_into_three)
		redist2_leaf(b, sib);

		/* choose which sibling to insert into */
		if (args->v->thin_begin < le64_to_cpu(sib_n->keys[0]))
			n2 = n;

		else {
			n2 = sib_n;
			index -= le32_to_cpu(n->header.nr_entries);
		}
		nr_entries = le32_to_cpu(n2->header.nr_entries);

		leaf_insert_(n2, args->v, index);

		res->nr_nodes = 2;
		init_node_info(&res->nodes[0], dm_block_location(b),
			       le64_to_cpu(n->keys[0]),
			       le32_to_cpu(n->header.nr_entries));
		init_node_info(&res->nodes[1], dm_block_location(sib),
			       le64_to_cpu(sib_n->keys[0]),
			       le32_to_cpu(sib_n->header.nr_entries));

		dm_tm_unlock(args->tm, sib);

		res->action[LEAF_SPLIT]++;
	} else {
		leaf_insert_(n, args->v, index);
		res->nr_nodes = 1;
		init_node_info(&res->nodes[0], dm_block_location(b),
			       le64_to_cpu(n->keys[0]),
			       le32_to_cpu(n->header.nr_entries));
	}

	res->flags = LEAF_NODE;

	return 0;
}

static void erase_from_leaf(struct leaf_node *node, unsigned index)
{
	uint32_t nr_entries = le32_to_cpu(node->header.nr_entries);
	if (index < nr_entries - 1) {
		array_erase(node->keys, sizeof(node->keys[0]), nr_entries,
			    index);
		array_erase(node->values, sizeof(node->values[0]), nr_entries,
			    index);
	}
	node->header.nr_entries = cpu_to_le32(nr_entries - 1);
}

static int action_to_left(struct dm_mapping *left, struct dm_mapping *right)
{
	int relation = test_relation(left, right);

	//printk(KERN_DEBUG "TL left %llu right %llu, r=%x", left->thin_begin, right->thin_begin, relation);

	// assume that right.len == 1, the relation falls into those categories,
	// in which OVERLAPPED_PARTIAL is not possible:
	switch (relation) {
	case (ADJACENTED | COMPATIBLE):
		if (left->len < MAPPINGS_MAX_LEN)
			return PUSH_BACK;
		else
			return INSERT_NEW;
	case (ADJACENTED | INCOMPATIBLE):
	case (DISJOINTED | INCOMPATIBLE):
	case (DISJOINTED | COMPATIBLE):
		return INSERT_NEW;
	case (OVERLAPPED_ALL | INCOMPATIBLE):
		return REPLACE;
	case (OVERLAPPED_HEAD | INCOMPATIBLE):
		return TRUNCATE_FRONT;
	case (OVERLAPPED_TAIL | INCOMPATIBLE):
		return TRUNCATE_BACK;
	case (OVERLAPPED_CENTER | INCOMPATIBLE):
		return SPLIT;
	default:
		return NOOP;	// incl. all other overlapped but compatible cases
	}
}

static int action_to_right(struct dm_mapping *left, struct dm_mapping *right)
{
	int relation = test_relation(left, right);

	//printk(KERN_DEBUG "TR left %llu right %llu, r=%x", left->thin_begin, right->thin_begin, relation);

	// assume that left.len == 1, the relation falls into these categories,
	// in which OVERLAPPED_PARTIAL is not possible:
	switch (relation) {
	case (ADJACENTED | COMPATIBLE):
		if (right->len < MAPPINGS_MAX_LEN)
			return PUSH_FRONT;
		else
			return INSERT_NEW;
	case (ADJACENTED | INCOMPATIBLE):
	case (DISJOINTED | INCOMPATIBLE):
	case (DISJOINTED | COMPATIBLE):
		return INSERT_NEW;
	default:
		return NOOP;	// not possible
	}
}

static void truncate_front(struct leaf_node *n, int i, uint32_t len)
{
	struct dm_mapping m;
	struct disk_mapping *value_le = n->values + i;
	uint64_t delta;

	get_mapping(n, i, &m);
	delta = (uint64_t) m.len - (uint64_t) len;	// cast to u64 to support underflow
	n->keys[i] = cpu_to_le64(m.thin_begin + delta);
	value_le->data_begin = cpu_to_le32(m.data_begin + delta);
	value_le->len_time = cpu_to_le32(pack_len_time(len, m.time));
}

// Similar to the case (e) in remove_leaf_()
static int remove_middle_(struct dm_transaction_manager *tm,
			  struct dm_space_map *data_sm, struct dm_block *b,
			  int i, uint64_t thin_begin, uint64_t thin_end,
			  struct insert_result *res)
{
	struct dm_mapping m;
	struct dm_mapping back_half;
	struct insert_args i_args;
	struct leaf_node *n;
	uint64_t front_half_len;
	uint64_t padding;
	int r;

	n = dm_block_data(b);
	get_mapping(n, i, &m);

	/* truncate the front half */
	front_half_len = thin_begin - m.thin_begin;
	set_len(n, i, front_half_len);

	/* insert new back half entry */
	padding = thin_end - m.thin_begin;

	back_half.thin_begin = thin_end;
	back_half.data_begin = m.data_begin + padding;
	back_half.len = m.len - padding;
	back_half.time = m.time;

	i_args.tm = tm;
	i_args.data_sm = data_sm;
	i_args.v = &back_half;

	r = insert_into_leaf(&i_args, b, i + 1, res);
	if (r)
		return r;

	return dm_sm_dec_blocks(data_sm, m.data_begin + front_half_len,
				back_half.data_begin);
}

static int overwrite_middle(struct insert_args *args, struct dm_block *b, int i,
			    struct insert_result *res)
{
	struct insert_result i_res;
	struct dm_block_manager *bm;
	bool overwrite_at_right = false;
	int r;

	// remove the middle part (might split)
	r = remove_middle_(args->tm, args->data_sm, b, i, args->v->thin_begin,
			   args->v->thin_begin + args->v->len, &i_res);
	if (r)
		return r;

	// choose which sibling to insert the overwritten key
	if (i_res.nr_nodes == 2
	    && args->v->thin_begin >= i_res.nodes[1].lowest_key) {
		bm = dm_tm_get_bm(args->tm);
		r = dm_bm_write_lock(bm, i_res.nodes[1].loc, &validator, &b);
		if (r)
			return r;
		i -= i_res.nodes[0].nr_entries;
		overwrite_at_right = true;
	}
	// do overwriting (might split if it didn't splitted previously)
	r = insert_into_leaf(args, b, i + 1, res);
	if (r) {
		if (overwrite_at_right)
			dm_bm_unlock(b);
		return r;
	}
	// return results
	if (overwrite_at_right) {
		res->nr_nodes = i_res.nr_nodes;
		res->nodes[1] = res->nodes[0];
		res->nodes[0] = i_res.nodes[0];
		dm_bm_unlock(b);
	} else if (i_res.nr_nodes == 2) {
		res->nr_nodes = i_res.nr_nodes;
		res->nodes[1] = i_res.nodes[1];
	}

	return 0;
}

static int leaf_insert(struct insert_args *args, struct dm_block *b,
		       struct insert_result *res)
{
	int i;
	struct leaf_node *n = dm_block_data(b);
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	struct dm_mapping center;
	struct dm_mapping left;
	struct dm_mapping right;
	struct dm_mapping *value = args->v;
	int action = 0;

	if (nr_entries == 0) {
		return insert_into_leaf(args, b, 0, res);
	}
	// FIXME: would this be better named 'index'
	i = lower_bound(n->keys, nr_entries, args->v->thin_begin);
	if (i >= 0 && i < nr_entries)
		get_mapping(n, i, &center);
	if (i > 0)
		get_mapping(n, i - 1, &left);
	if (i + 1 < nr_entries)
		get_mapping(n, i + 1, &right);

	// currently support inserting len-1 ranges only
	if (i < 0) {
		action = action_to_right(value, &right);
	} else {
		action = action_to_left(&center, value);
		switch (action) {
		case REPLACE:
			if (i > 0 && action_to_left(&left, value) == PUSH_BACK) {
				if (i + 1 < nr_entries &&
				    action_to_right(value, &right) == PUSH_FRONT
				    && left.len + right.len < MAPPINGS_MAX_LEN)
					action = MERGE_REPLACED;
				else
					// prefers PUSH_BACK_REPLACED rather than
					// PUSH_FRONT_REPLACED if the merged length overflows
					action = PUSH_BACK_REPLACED;
			} else if (i + 1 < nr_entries &&
				   action_to_right(value, &right) == PUSH_FRONT)
				action = PUSH_FRONT_REPLACED;
			break;
		case TRUNCATE_FRONT:
			if (i > 0 && action_to_left(&left, value) == PUSH_BACK)
				action = PUSH_BACK_TRUNCATED;
			break;
		case TRUNCATE_BACK:
			if (i + 1 < nr_entries
			    && action_to_right(value, &right) == PUSH_FRONT)
				action = PUSH_FRONT_TRUNCATED;
			break;
		case PUSH_BACK:
			if (i + 1 < nr_entries &&
			    action_to_right(value, &right) == PUSH_FRONT &&
			    center.len + right.len < MAPPINGS_MAX_LEN)
				action = MERGE;
			break;
		case INSERT_NEW:
			if (i + 1 < nr_entries)
				action = action_to_right(value, &right);
			break;
		}
	}

	//printk(KERN_DEBUG "key = %llu, i = %d, v = %llu, action = %u", args->v->thin_begin, i, args->v->data_begin, action);

	switch (action) {
	case PUSH_BACK:
		set_len(n, i, center.len + 1);
		break;
	case PUSH_BACK_REPLACED:
		erase_from_leaf(n, i);
		set_len(n, i - 1, left.len + 1);
		break;
	case PUSH_BACK_TRUNCATED:
		truncate_front(n, i, center.len - 1);
		set_len(n, i - 1, left.len + 1);
		break;
	case PUSH_FRONT:
		truncate_front(n, i + 1, right.len + 1);
		break;
	case PUSH_FRONT_REPLACED:
		erase_from_leaf(n, i);
		truncate_front(n, i, right.len + 1);
		break;
	case PUSH_FRONT_TRUNCATED:
		set_len(n, i, center.len - 1);
		truncate_front(n, i + 1, right.len + 1);
		break;
	case MERGE:
		erase_from_leaf(n, i + 1);
		set_len(n, i, center.len + right.len + 1);
		break;
	case MERGE_REPLACED:
		erase_from_leaf(n, i);	// FIXME: do not memmove twice
		erase_from_leaf(n, i);
		set_len(n, i - 1, left.len + right.len + 1);
		break;
	case TRUNCATE_BACK:
		set_len(n, i, center.len - 1);
		return insert_into_leaf(args, b, i + 1, res);
	case TRUNCATE_FRONT:
		truncate_front(n, i, center.len - 1);
		return insert_into_leaf(args, b, i, res);
	case REPLACE:
		erase_from_leaf(n, i);	// FIXME: do not erase to avoid memmove
		return insert_into_leaf(args, b, i, res);
	case INSERT_NEW:
		return insert_into_leaf(args, b, i + 1, res);
	case SPLIT:
		return overwrite_middle(args, b, i, res);
	}

	res->nr_nodes = 1;
	res->nodes[0].loc = dm_block_location(b);
	res->nodes[0].lowest_key = le64_to_cpu(n->keys[0]);
	res->nodes[0].nr_entries = le32_to_cpu(n->header.nr_entries);
	res->flags = LEAF_NODE;

	return 0;
}

static struct node_ops leaf_ops = {
	.rebalance2 = leaf_rebalance2,
	.rebalance3 = leaf_rebalance3,
	.del = leaf_del,
	.insert = leaf_insert,
	.remove = NULL
};

static int get_ops(struct node_header *h, struct node_ops **ops)
{
	uint32_t flags = le32_to_cpu(h->flags);
	if (flags == LEAF_NODE)
		*ops = &leaf_ops;
	else if (flags == INTERNAL_NODE)
		*ops = &internal_ops;
	else
		return -EINVAL;
	return 0;
}

/*----------------------------------------------------------------*/

int dm_rtree_empty(struct dm_transaction_manager *tm, dm_block_t *root)
{
	int r;
	struct dm_block *b;
	struct node_header *h;

	r = dm_tm_new_block(tm, &validator, &b);
	if (r < 0)
		return r;

	h = dm_block_data(b);
	h->flags = cpu_to_le32(LEAF_NODE);
	h->blocknr = cpu_to_le64(dm_block_location(b));
	h->nr_entries = cpu_to_le32(0);

	*root = dm_block_location(b);
	dm_tm_unlock(tm, b);

	return 0;
}

EXPORT_SYMBOL_GPL(dm_rtree_empty);

/*----------------------------------------------------------------*/

int dm_rtree_del(struct dm_transaction_manager *tm,
		 struct dm_space_map *data_sm, dm_block_t root)
{
	struct del_args args = {.tm = tm,.data_sm = data_sm };
	return del_(&args, root);

}

EXPORT_SYMBOL_GPL(dm_rtree_del);

/*----------------------------------------------------------------*/

int dm_rtree_lookup(struct dm_transaction_manager *tm, dm_block_t root,
		    dm_block_t key, struct dm_mapping *result)
{
	int i, r;
	bool found = false;
	struct dm_block *b;
	uint32_t flags, nr_entries;
	struct node_header *h;
	struct internal_node *n;

	while (!found) {
		r = dm_tm_read_lock(tm, root, &validator, &b);
		if (r < 0)
			return r;

		h = dm_block_data(b);
		flags = le32_to_cpu(h->flags);
		nr_entries = le32_to_cpu(h->nr_entries);

		n = (struct internal_node *)h;
		i = lower_bound(n->keys, nr_entries, key);
		if (i < 0 || i >= nr_entries) {
			dm_tm_unlock(tm, b);
			return -ENODATA;
		}

		if (flags & INTERNAL_NODE)
			root = le64_to_cpu(n->values[i]);

		else {
			dm_block_t thin_end;
			struct leaf_node *n = (struct leaf_node *)h;

			get_mapping(n, i, result);
			thin_end = result->thin_begin + result->len;

			if (key > thin_end) {
				dm_tm_unlock(tm, b);
				return -ENODATA;
			}

			found = true;
		}

		dm_tm_unlock(tm, b);
	}

	return r;
}

EXPORT_SYMBOL_GPL(dm_rtree_lookup);

/*----------------------------------------------------------------*/

#if 0
/*
 * We often need to modify a sibling node.  This function shadows a particular
 * child of the given parent node.  Making sure to update the parent to point
 * to the new shadow.
 */
static int shadow_child(struct dm_transaction_manager *tm,
			struct dm_space_map *data_sm,
			struct internal_node *pn,
			unsigned index, struct dm_block **result)
{
	int r, inc;
	dm_block_t block = le64_to_cpu(pn->values[index]);
	r = dm_tm_shadow_block(tm, block, &validator, result, &inc);
	if (r)
		return r;

	if (inc)
		inc_children(tm, data_sm, *result);

	pn->values[index] = cpu_to_le64(dm_block_location(*result));
	return 0;
}
#endif

/*
 * Returns the number of spare entries in a node.
 */
static int get_node_free_space(struct dm_transaction_manager *tm, dm_block_t b,
			       unsigned *space)
{
	int r;
	unsigned nr_entries;
	struct dm_block *block;
	struct node_header *h;

	r = dm_tm_read_lock(tm, b, &validator, &block);
	if (r)
		return r;

	h = dm_block_data(block);
	nr_entries = le32_to_cpu(h->nr_entries);
	if (le32_to_cpu(h->flags) & INTERNAL_NODE)
		*space = INTERNAL_NR_ENTRIES - nr_entries;
	else
		*space = LEAF_NR_ENTRIES - nr_entries;

	dm_tm_unlock(tm, block);
	return 0;
}

int dm_rtree_insert(struct dm_transaction_manager *tm,
		    struct dm_space_map *data_sm,
		    dm_block_t root,
		    struct dm_mapping *value, dm_block_t *new_root,
		    unsigned *nr_inserts)
{
	int r;
	struct insert_args args = {.tm = tm,.data_sm = data_sm,.v = value };
	struct insert_result res = { 0 };
	r = insert_aux(&args, root, &res);
	if (r)
		return r;

	if (res.nr_nodes == 1) {
		*new_root = res.nodes[0].loc;

	} else {
		/* we need to create a new layer */
		struct dm_block *b;
		struct internal_node *n;

		r = dm_tm_new_block(tm, &validator, &b);
		if (r)
			return r;

		n = dm_block_data(b);
		n->header.flags = cpu_to_le32(INTERNAL_NODE);
		n->header.nr_entries = 2;
		n->keys[0] = cpu_to_le64(res.nodes[0].lowest_key);
		n->values[0] = cpu_to_le64(res.nodes[0].loc);
		n->keys[1] = cpu_to_le64(res.nodes[1].lowest_key);
		n->values[1] = cpu_to_le64(res.nodes[1].loc);

		*new_root = dm_block_location(b);
		dm_tm_unlock(tm, b);
	}

	/*for (i = 1; i <= 6; i++) {
	   if (res.action[i] > 0)
	   break;
	   }
	   for (j = 1; j <= 5; j++) {
	   if (res.stats[j] > 0)
	   break;
	   }
	   if (i <= 6 || j <= 5) {
	   printk(KERN_DEBUG "leaf: 0 s%u o%u u%u", res.action[LEAF_SPLIT], res.action[LEAF_OVERFLOW], res.action[LEAF_UNDERFLOW]);
	   printk(KERN_DEBUG "internal: s%u o%u u%u", res.action[INTERNAL_SPLIT], res.action[INTERNAL_OVERFLOW], res.action[INTERNAL_UNDERFLOW]);
	   printk(KERN_DEBUG "0 %u 2r%u 3r%u", res.stats[NOTHING], res.stats[REDIST2], res.stats[REDIST3]);
	   printk(KERN_DEBUG "1m%u 2m%u", res.stats[MERGED_INTO_ONE], res.stats[MERGED_INTO_TWO]);
	   } */

	dm_tm_update_stats(tm, res.action, res.stats);

	return 0;
}

EXPORT_SYMBOL_GPL(dm_rtree_insert);

/*----------------------------------------------------------------*/

//#if 0
static int remove_(struct dm_transaction_manager *tm,
		   struct dm_space_map *data_sm,
		   dm_block_t b,
		   dm_block_t thin_begin, dm_block_t thin_end,
		   struct remove_result *res);

static void erase_internal_entry(struct internal_node *n, unsigned index)
{
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	size_t upper = nr_entries - index - 1;
	n->header.nr_entries = cpu_to_le32(nr_entries - 1);
	if (upper > 0) {
		memmove(n->keys + index, n->keys + index + 1,
			sizeof(n->keys[0]) * upper);
		memmove(n->values + index, n->values + index + 1,
			sizeof(n->values[0]) * upper);
	}
}

static void erase_leaf_entries(struct leaf_node *n, unsigned index_b,
			       unsigned index_e)
{
	size_t upper = le32_to_cpu(n->header.nr_entries) - index_e;
	n->header.nr_entries = cpu_to_le32(index_b + upper);
	if (upper > 0) {
		memmove(n->keys + index_b, n->keys + index_e,
			sizeof(n->keys[0]) * upper);
		memmove(n->values + index_b, n->values + index_e,
			sizeof(n->values[0]) * upper);
	}
}

/*
 * We make no attempt to rebalance the nodes after the remove.  Now that
 * we're removing ranges it just gets too complicated.  Instead we'll track
 * the average residency (nr_entries, nr_internal, nr_leaf) and rebuild the
 * tree if it gets too bad.
 * 
 * If parent is 0, then there is no parent and block is the root of the tree.
 */
// FIXME: we don't need the new_root param
static int remove_internal_(struct dm_transaction_manager *tm,
			    struct dm_space_map *data_sm,
			    struct dm_block *block,
			    dm_block_t thin_begin, dm_block_t thin_end,
			    struct remove_result *res)
{
	struct internal_node *n = dm_block_data(block);
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);
	int r, i = lower_bound(n->keys, nr_entries, thin_begin);
	if (i < 0)
		i = 0;

	res->nr_nodes = 0;

	for (; i < nr_entries; i++) {
		dm_block_t key = le64_to_cpu(n->keys[i]);
		dm_block_t next_key;
		dm_block_t child;

		if (key >= thin_end)
			break;

		if (i == (nr_entries - 1)) {
			child = le64_to_cpu(n->values[i]);

			// FIXME: node_end is error prone, so I'm going to just recurse for now.
			// FIXME: Do not recurse if key >= thin_end ?
			r = remove_(tm, data_sm, child,
				    thin_begin, thin_end, res);
			if (r)
				return r;

			/* Remove or update the child pointer */
			if (res->nodes[0].nr_entries == 0) {
				erase_internal_entry(n, i);
				i -= 1;
				nr_entries -= 1;
				dm_tm_dec(tm, res->nodes[0].loc);
			} else {
				n->values[i] = cpu_to_le64(res->nodes[0].loc);
			}
		} else {
			next_key = le64_to_cpu(n->keys[i + 1]);

			if (next_key <= thin_begin)
				continue;

			if (key >= thin_begin && next_key <= thin_end) {
				/* the whole entry is within the removal range, so we hand over to 
				 * rtree_del.  FIXME: this mean _del can't call kmalloc, rewrite using
				 * stack/recursion.
				 */
				// FIXME: repeated moves
				child = le64_to_cpu(n->values[i]);
				erase_internal_entry(n, i);
				i -= 1;
				nr_entries -= 1;
				{
					r = dm_rtree_del(tm, data_sm, child);
					if (r) {
						return r;
					}
				}
			} else {
				child = le64_to_cpu(n->values[i]);

				/* There's an overlap, recurse into the child */
				r = remove_(tm, data_sm, child,
					    thin_begin, thin_end, res);
				if (r)
					return r;

				/* Remove or update the child pointer */
				if (res->nodes[0].nr_entries == 0) {
					erase_internal_entry(n, i);
					i -= 1;
					nr_entries -= 1;
					dm_tm_dec(tm, res->nodes[0].loc);
				} else {
					n->values[i] =
					    cpu_to_le64(res->nodes[0].loc);
				}
			}
		}
	}

	// in-place modification:
	//   res1 == 0, res2 == 0  // no overlap. all are removed
	//   res1 == 0, res2 == 1  // lower_bound() == thin_begin; overlaps to the thin_end
	//   res1 == 1, res2 == 0  // overlaps to the begin; last key is removed
	//   res1 == 1, res2 == 1  // normal case
	// may split:
	//   res1 == 0, res2 == 2
	//   res1 == 2, res2 == 0

	if (res->nr_nodes <= 1) {
		dm_block_t node_end;

		res->nr_nodes = 1;
		init_node_info(&res->nodes[0], dm_block_location(block),
			       le64_to_cpu(n->keys[0]),
			       le32_to_cpu(n->header.nr_entries));

		/* adjust the node_end */
		node_end = le64_to_cpu(n->header.node_end);
		if ((thin_begin < node_end) && (thin_end >= node_end))
			n->header.node_end = cpu_to_le64(thin_begin);
	} else {
		/* One of the children was splitted into two. Insert the newly allocated node */
		struct node_info *new_node;
		new_node = &res->nodes[1];

		if (nr_entries < INTERNAL_NR_ENTRIES) {
			dm_block_t node_end;

			insert_into_internal(n, i, new_node->lowest_key,
					     new_node->loc);

			/* adjust the node_end */
			node_end = le64_to_cpu(n->header.node_end);
			if ((thin_begin < node_end) && (thin_end >= node_end))
				n->header.node_end = cpu_to_le64(thin_begin);

			res->nr_nodes = 1;
			init_node_info(&res->nodes[0], dm_block_location(block),
				       le64_to_cpu(n->keys[0]),
				       le32_to_cpu(n->header.nr_entries));
		} else {
			/* split the node */
			struct dm_block *sib;
			struct internal_node *sib_n, *n2;

			r = dm_tm_new_block(tm, &validator, &sib);
			if (r < 0)
				return r;

			sib_n = dm_block_data(sib);

			sib_n->header.flags = n->header.flags;
			sib_n->header.nr_entries = cpu_to_le32(0);

			redist2_internal(block, sib);

			if (new_node->lowest_key < le64_to_cpu(sib_n->keys[0]))
				n2 = n;
			else {
				n2 = sib_n;
				i -= le32_to_cpu(n->header.nr_entries);
			}
			insert_into_internal(n2, i, new_node->lowest_key,
					     new_node->loc);

			// FIXME: adjust the node_end for the two nodes

			res->nr_nodes = 2;
			init_node_info(&res->nodes[0], dm_block_location(block),
				       le64_to_cpu(n->keys[0]),
				       le32_to_cpu(n->header.nr_entries));
			init_node_info(&res->nodes[1], dm_block_location(sib),
				       le64_to_cpu(sib_n->keys[0]),
				       le32_to_cpu(sib_n->header.nr_entries));

			dm_tm_unlock(tm, sib);
		}
	}

	// FIXME: ugly
	if (n->header.nr_entries == 0) {
		n->header.flags = LEAF_NODE;
	} else {
		// TODO: run rebalancing?
		// 1. determine the rebalancing index
		// 2. do rebalancing
	}

	return 0;
}

/*
 * Lot's of cases here.  An entry can be classified in relation to
 * thin_begin and thin_end:
 * 
 *   a) entirely below (many)
 *   b) overlap begin (single)
 *   c) entirely within (many)
 *   d) overlap end (single)
 *   e) overlap begin and end (single)
 *   f) entirely above (many)
 * 
 * Moving keys and values around is expensive, so we'd like to remove
 * (c) cases all in one operation.
 * 
 * (e) will cause an entry to be split into two smaller entries.  So removing
 * a range can increase the number of entries in the leaf, so we have to ensure
 * there's space before calling this.
 */

static int remove_leaf_(struct dm_transaction_manager *tm,
			struct dm_space_map *data_sm,
			struct dm_block *block,
			dm_block_t thin_begin, dm_block_t thin_end,
			struct remove_result *res)
{
	int i, r;
	unsigned within_start, within_end;
	struct disk_mapping *value_le;
	struct dm_mapping m;
	struct leaf_node *n = dm_block_data(block);
	uint32_t nr_entries = le32_to_cpu(n->header.nr_entries);

	if (nr_entries == 0) {
		return 0;
	}

	i = lower_bound(n->keys, nr_entries, thin_begin);
	if (i < 0)
		i = 0;

	value_le = n->values + i;
	get_mapping(n, i, &m);

	if (m.thin_begin < thin_begin && (m.thin_begin + m.len) > thin_end) {
		/* case e */
		struct insert_result insert_res;
		remove_middle_(tm, data_sm, block, i, thin_begin, thin_end,
			       &insert_res);
		res->nr_nodes = insert_res.nr_nodes;
		memcpy(res->nodes, insert_res.nodes, sizeof(insert_res.nodes));
	} else {
		if (m.thin_begin + m.len <= thin_begin) {
			/* case a */
			i++;
		} else if (m.thin_begin < thin_begin) {
			/* case b */
			dm_block_t delta = thin_begin - m.thin_begin;
			dm_block_t data_begin = m.data_begin + delta;
			dm_block_t data_end = m.data_begin + m.len;
			r = dm_sm_dec_blocks(data_sm, data_begin, data_end);
			if (r)
				return r;
			set_len(n, i, thin_begin - m.thin_begin);
			i++;
		}

		/* Collect entries that are entirely within the remove range */
		within_start = i;
		for (; i < nr_entries; i++) {
			get_mapping(n, i, &m);

			if (m.thin_begin + m.len > thin_end)
				break;

			/* case c */
			{
				dm_block_t data_end = m.data_begin + m.len;
				r = dm_sm_dec_blocks(data_sm, m.data_begin,
						     data_end);
				if (r)
					return r;
			}
		}
		within_end = i;

		if (within_end > within_start) {
			unsigned shift = within_end - within_start;
			erase_leaf_entries(n, within_start, within_end);
			nr_entries -= shift;
			i -= shift;
		}

		if (i < nr_entries) {
			get_mapping(n, i, &m);

			if (m.thin_begin < thin_end) {
				/* case d */
				dm_block_t delta;

				delta = thin_end - m.thin_begin;
				r = dm_sm_dec_blocks(data_sm, m.data_begin,
						     m.data_begin + delta);
				if (r)
					return r;
				truncate_front(n, i, m.len - delta);
			}
		}

		res->nr_nodes = 1;
		init_node_info(res->nodes, dm_block_location(block),
			       le64_to_cpu(n->keys[0]),
			       le32_to_cpu(n->header.nr_entries));
	}

	/*
	 * We need to reread nr_entries, since erase ops from above may have
	 * changed it.
	 */
	nr_entries = le32_to_cpu(n->header.nr_entries);

	/* adjust the node_end, which may have changed */
	if (nr_entries) {
		struct disk_mapping *value_le = n->values + nr_entries - 1;
		uint32_t len;
		uint32_t time;
		unpack_len_time(le32_to_cpu(value_le->len_time), &len, &time);
		n->header.node_end =
		    cpu_to_le64(le64_to_cpu(n->keys[nr_entries - 1]) + len);
	}

	return 0;
}

// FIXME: removing the middle of a mapping can cause an extra entry to
// be inserted.  So we need to ensure there's enough space.
static int remove_(struct dm_transaction_manager *tm,
		   struct dm_space_map *data_sm,
		   dm_block_t b,
		   dm_block_t thin_begin, dm_block_t thin_end,
		   struct remove_result *res)
{
	int r, inc;
	struct node_header *h;
	struct dm_block *block;

	r = dm_tm_shadow_block(tm, b, &validator, &block, &inc);
	if (r)
		return r;

	if (inc)
		inc_children(tm, data_sm, block);

	h = dm_block_data(block);

	if (le32_to_cpu(h->flags) & INTERNAL_NODE)
		r = remove_internal_(tm, data_sm, block, thin_begin, thin_end,
				     res);
	else
		r = remove_leaf_(tm, data_sm, block, thin_begin, thin_end, res);

	dm_tm_unlock(tm, block);
	return r;
}

//#endif

// FIXME: we need to know how many mappings were removed.
int dm_rtree_remove(struct dm_transaction_manager *tm,
		    struct dm_space_map *data_sm,
		    dm_block_t b,
		    dm_block_t thin_begin, dm_block_t thin_end,
		    dm_block_t *new_root)
{
	int r;
	struct remove_result res;

	r = remove_(tm, data_sm, b, thin_begin, thin_end, &res);
	if (r)
		return r;

	if (res.nr_nodes == 1) {
		*new_root = res.nodes[0].loc;
	} else {
		/* we need to create a new layer */
		struct dm_block *b;
		struct internal_node *n;

		r = dm_tm_new_block(tm, &validator, &b);
		if (r)
			return r;

		n = dm_block_data(b);
		n->header.flags = cpu_to_le32(INTERNAL_NODE);
		n->header.nr_entries = 2;
		n->keys[0] = cpu_to_le64(res.nodes[0].lowest_key);
		n->values[0] = cpu_to_le64(res.nodes[0].loc);
		n->keys[1] = cpu_to_le64(res.nodes[1].lowest_key);
		n->values[1] = cpu_to_le64(res.nodes[1].loc);

		*new_root = dm_block_location(b);
		dm_tm_unlock(tm, b);
	}

	return 0;
}

EXPORT_SYMBOL_GPL(dm_rtree_remove);

int dm_rtree_find_highest_key(struct dm_transaction_manager *tm,
			      dm_block_t root, dm_block_t *thin_block_result)
{
	return -EINVAL;
}

EXPORT_SYMBOL_GPL(dm_rtree_find_highest_key);

/*----------------------------------------------------------------*/
