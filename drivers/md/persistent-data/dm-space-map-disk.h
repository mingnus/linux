/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (C) 2011 Red Hat, Inc.
 *
 * This file is released under the GPL.
 */

#ifndef _LINUX_DM_SPACE_MAP_DISK_H
#define _LINUX_DM_SPACE_MAP_DISK_H

#include "dm-block-manager.h"

struct dm_space_map;
struct dm_transaction_manager;

/*
 * Unfortunately we have to use two-phase construction due to the cycle
 * between the tm and sm.
 */
struct dm_space_map *dm_sm_disk_create(struct dm_transaction_manager *tm,
				       dm_block_t nr_blocks);

struct dm_space_map *dm_sm_disk_open(struct dm_transaction_manager *tm,
				     void *root, size_t len);

/*
 * Fence off [begin, end) from new_block, eg. while free blocks within the
 * region are being discarded.  Only one window may be active at a time;
 * -EBUSY is returned otherwise.  The caller must serialise these calls
 * against new_block.
 */
int dm_sm_disk_set_exclusion(struct dm_space_map *sm, dm_block_t begin,
			     dm_block_t end);
void dm_sm_disk_clear_exclusion(struct dm_space_map *sm);

/*
 * Find the next run of blocks within [begin, end) that is free in both the
 * current and the last committed transaction.  Does not allocate.
 */
int dm_sm_disk_next_free_run(struct dm_space_map *sm, dm_block_t begin,
			     dm_block_t end, dm_block_t *result_begin,
			     dm_block_t *result_end);

#endif /* _LINUX_DM_SPACE_MAP_DISK_H */
