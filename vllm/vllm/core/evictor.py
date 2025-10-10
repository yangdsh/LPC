# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import time
import statistics
import numpy as np
from collections import defaultdict
from sortedcontainers import SortedDict
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

def probability_of_future_arrival(prob_has_next, exp_scale, elapsed_time, debug=False):
    if prob_has_next == 0 or exp_scale == 0:
        return 0.0
    prob_not_accessed_till_now = np.exp(-elapsed_time / exp_scale)
    return (prob_has_next * prob_not_accessed_till_now) / (
            prob_has_next * prob_not_accessed_till_now + (1 - prob_has_next)
    )

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass

class CacheStat:
    def __init__(self):
        self.stat = defaultdict(list)
        self.average = {}
        self.last_log_time = 0
        self.log_interval = 10000  # Log every 5 seconds
    
    def get_average(self, key):
        if key in self.average:
            return self.average[key]
        else:
            return 1
    
    def append(self, key, value):
        if key not in self.average:
            self.average[key] = 0
        self.average[key] = (self.average[key] * len(self.stat[key]) + value) / (len(self.stat[key]) + 1)
        self.stat[key].append(value)
    
    def summary(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return 0
        self.last_log_time = current_time
        has_data = 0

        for key in self.stat.keys():
            data = self.stat[key]
            if not data:
                continue
            
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0
            print(f"Summary for {key}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
            has_data = 1
        return has_data
        

class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, cache_hint: dict = None, score: float = 0):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.cache_hint = cache_hint
        self.score = score

class LRUMLEvictor(Evictor):

    def __init__(self, config):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.sorted_dict = SortedDict()
        self.id_to_last_access = {}
        self.id_to_first_access = {}
        self.to_delete_blocks = []
        self.config = self.parse_str_to_dict(config)
        self.stat = CacheStat()
        self.last_refresh_time = time.time()
        self.INSPECT_INTERVAL = 5

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def parse_str_to_dict(self, s: str) -> dict:
        if len(s) == 0:
            return {}
        return {key: value for key, value in (pair.split("=", 1) for pair in s.split(","))}

    def calc_score(self, block_id, last_accessed, cache_hint):
        if 'use_lru' in cache_hint and cache_hint['use_lru']:
            return last_accessed
        if 'use_fifo' in cache_hint and cache_hint['use_fifo']:
            return self.id_to_first_access[block_id]
        if 'next_timestamp' in cache_hint:
            return -cache_hint['next_timestamp']
        if 'prob_has_next' in cache_hint:
            prob = probability_of_future_arrival(
                cache_hint['prob_has_next'], cache_hint['exp_scale'], time.time() - last_accessed)
            return prob

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        block_id = -1
        while block_id not in self.free_table:
            if len(self.to_delete_blocks) > 0:
                (block_id, content_hash, last_accessed) = self.to_delete_blocks.pop()
                if block_id not in self.free_table or self.free_table[block_id].last_accessed != last_accessed:
                    continue
            else:
                _, (block_id, content_hash) = self.sorted_dict.popitem(0)
        if block_id in self.free_table:
            survival_time = time.time() - self.free_table[block_id].last_accessed
            self.stat.append("survival_times", survival_time)
            del self.free_table[block_id]
            del self.id_to_first_access[block_id]
            return block_id, content_hash
        else:
            # print('block is not in the sorted_dict')
            raise ValueError("block is not in the sorted_dict")
    
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        score = self.calc_score(block_id, last_accessed, cache_hint)
        # print("add: ", block_id, cache_hint)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint,
                                                  score)
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed
        if block_id not in self.id_to_first_access:
            self.id_to_first_access[block_id] = last_accessed
        if time.time() - self.last_refresh_time > self.INSPECT_INTERVAL:
            self._refresh()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            raise ValueError("Attempting to update block that's not in the evictor")
        print("update: ", block_id, cache_hint['turns'])
        old_score = self.free_table[block_id].score
        old_entry = (old_score, self.free_table[block_id].last_accessed, block_id)
        if old_entry in self.sorted_dict:
            del self.sorted_dict[old_entry]
        else:
            raise ValueError("the score is not found in sorted_dict")
        
        score = self.calc_score(block_id, last_accessed, cache_hint)
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint
        self.free_table[block_id].score = score
        
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, self.free_table[block_id].content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed

    # remove is only called by 'hit', the blocks will be added back later after decoding
    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        
        # print("remove: ", block_id)
        old_score = self.free_table[block_id].score
        old_entry = (old_score, self.free_table[block_id].last_accessed, block_id)
        if old_entry in self.sorted_dict:
            del self.sorted_dict[old_entry]
        else:
            raise ValueError("the score is not found in sorted_dict")
        del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def _refresh(self):
        new_sorted_dict = SortedDict()

        for block_id, block in self.free_table.items():
            score = self.calc_score(block_id, block.last_accessed, block.cache_hint)
            block.score = score
            new_sorted_dict[(score, block.last_accessed, block_id)] = (block_id, block.content_hash)

        self.sorted_dict = new_sorted_dict

        print('num blocks: ', self.num_blocks)

        self.last_refresh_time = time.time()
        stat_ = CacheStat()
        # Create a list of (block_id, survival_time) tuples
        survival_list = [
            (block_id, self.free_table[block_id].cache_hint['id'], self.free_table[block_id].last_accessed)
            for block_id in self.free_table
        ]
        # Sort by survival_time in descending order
        survival_list.sort(key=lambda x: (x[2], x[1]))

        # mark outdated blocks 
        to_delete_cnt = 0
        for block_id, _, _ in survival_list:
            block = self.free_table[block_id]
            id = block.cache_hint['id']
            if self.id_to_last_access[id] == block.last_accessed:
                continue
            if self.id_to_last_access[id] != block.last_accessed:
                self.to_delete_blocks.append((block_id, block.content_hash, block.last_accessed))
                to_delete_cnt += 1
        print("mark outdated blocks cnt: ", to_delete_cnt)

        # Print top 10
        cnt = 0
        temp = 0
        print('Top 10 blocks with oldest last_accessed time:')
        for block_id, _, _ in survival_list:
            block = self.free_table[block_id]
            if block.last_accessed == temp:
                continue
            temp = block.last_accessed
            cnt += 1
            if cnt > 10:
                break
            print(block.cache_hint['true_tta'], block.score, block.last_accessed)

        for block_id in self.free_table:
            survival_time = time.time() - self.free_table[block_id].last_accessed
            stat_.append("survival_times", survival_time)
        print('For all blocks in the cache:')
        stat_.summary()

class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.stat = CacheStat()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                survival_time = time.time() - self.free_table[block_id].last_accessed
                self.stat.append("survival_times", survival_time)
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if 'use_fifo' not in cache_hint or cache_hint['use_fifo'] == 0:
            self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_algorithm: str, config: str) -> Evictor:
    if eviction_algorithm == 'lru':
        return LRUEvictor()
    else:
        return LRUMLEvictor(config)
