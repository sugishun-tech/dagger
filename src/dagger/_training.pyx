# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language = c++
"""Independent feature-sequence pattern extraction, following Yoshinaga (2023).

The corpus is traversed with Cython loops.  Surface prefixes and observations
are interned in integer-indexed C++ containers: an occurrence does not allocate
a Python substring or a Counter for every right-context length.
"""

from libc.stdint cimport uint32_t, uint64_t, int32_t
from cpython.unicode cimport PyUnicode_READ_CHAR, PyUnicode_GET_LENGTH
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
from cython.operator cimport dereference as deref, preincrement as inc


cdef class _SurfaceTrie:
    cdef unordered_map[uint64_t, uint32_t] edges
    cdef vector[uint32_t] parent
    cdef vector[uint32_t] character

    def __cinit__(self):
        self.parent.push_back(0)
        self.character.push_back(0)

    cdef uint32_t extend(self, uint32_t node, uint32_t cp) except *:
        cdef uint64_t key = (<uint64_t>node << 32) | cp
        cdef unordered_map[uint64_t, uint32_t].iterator it = self.edges.find(key)
        cdef uint32_t child
        if it != self.edges.end():
            return deref(it).second
        if self.parent.size() >= 0xffffffff:
            raise OverflowError("too many training trie nodes")
        child = <uint32_t>self.parent.size()
        self.parent.push_back(node)
        self.character.push_back(cp)
        self.edges[key] = child
        return child

    cdef uint32_t insert(self, unicode surface) except *:
        cdef Py_ssize_t j, length = PyUnicode_GET_LENGTH(surface)
        cdef uint32_t node = 0
        for j in range(length):
            node = self.extend(node, PyUnicode_READ_CHAR(surface, j))
        return node

    cdef unicode surface(self, uint32_t node):
        cdef list chars = []
        while node:
            chars.append(chr(self.character[node]))
            node = self.parent[node]
        chars.reverse()
        return "".join(chars)


cdef tuple _check_token(object item):
    if not isinstance(item, (tuple, list)) or len(item) != 4:
        raise ValueError("a token must be (surface, pos, lemma, reading)")
    if not all(isinstance(field, str) for field in item):
        raise TypeError("token fields must be str")
    if not item[0] or not item[1]:
        raise ValueError("surface and POS must not be empty")
    if "\n" in item[0] or "\r" in item[0]:
        raise ValueError("a token surface must not cross a line boundary")
    return tuple(item)


cdef uint32_t _intern_action(
        tuple token, dict pos_ids, list pos_names, dict action_ids,
        list actions, vector[uint32_t]& shifts, vector[uint32_t]& pos_of) except *:
    cdef unicode surface = token[0]
    cdef unicode lemma = token[2] if token[2] else "*"
    cdef Py_ssize_t n = len(surface), common = 0, stop = min(n, len(lemma))
    cdef uint32_t pos_id, action_id
    cdef object found, key
    found = pos_ids.get(token[1])
    if found is None:
        if len(pos_names) >= 0xfffffffe:
            raise OverflowError("too many POS labels")
        pos_id = <uint32_t>len(pos_names)
        pos_ids[token[1]] = pos_id
        pos_names.append(token[1])
    else:
        pos_id = found
    while common < stop and PyUnicode_READ_CHAR(surface, common) == PyUnicode_READ_CHAR(lemma, common):
        common += 1
    key = (n, pos_id, n - common, lemma[common:], token[3] or "*")
    found = action_ids.get(key)
    if found is not None:
        return <uint32_t>found
    if len(actions) >= 0x7fffffff:
        raise OverflowError("too many morphological output labels")
    action_id = <uint32_t>len(actions)
    action_ids[key] = action_id
    actions.append((token[1], n - common, lemma[common:], token[3] or "*"))
    shifts.push_back(n)
    pos_of.push_back(pos_id)
    return action_id


cdef uint32_t _candidate(
        uint32_t node, uint32_t context,
        unordered_map[uint64_t, uint32_t]& candidates,
        vector[uint64_t]& candidate_keys) except *:
    cdef uint64_t key = (<uint64_t>node << 32) | context
    cdef unordered_map[uint64_t, uint32_t].iterator it = candidates.find(key)
    cdef uint32_t result
    if it != candidates.end():
        return deref(it).second
    if candidate_keys.size() >= 0x7fffffff:
        raise OverflowError("too many pattern candidates")
    result = <uint32_t>candidate_keys.size()
    candidate_keys.push_back(key)
    candidates[key] = result
    return result


cdef uint32_t _choose_action(
        vector[pair[uint64_t, uint64_t]]& records,
        size_t start, size_t end,
        vector[uint32_t]& shifts, vector[uint32_t]& pos_of,
        list actions, list pos_names, uint64_t* total) except *:
    """Marginalize shift, then POS; choose metadata only within the winning POS."""
    cdef size_t j
    cdef uint32_t action, shift, pos, best_shift = 0, best_pos = 0, best_action = 0
    cdef uint64_t count, best_count = 0
    cdef dict shift_counts = {}, pos_counts = {}
    cdef object value, pos_name
    total[0] = 0
    if end == start + 1:
        total[0] = records[start].second
        return <uint32_t>records[start].first
    for j in range(start, end):
        action = <uint32_t>records[j].first
        count = records[j].second
        total[0] += count
        shift = shifts[action]
        shift_counts[shift] = shift_counts.get(shift, 0) + count
    # Prefer the longer segmentation offset on a frequency tie.
    for value, count in shift_counts.items():
        shift = value
        if count > best_count or (count == best_count and shift > best_shift):
            best_count = count
            best_shift = shift
    for j in range(start, end):
        action = <uint32_t>records[j].first
        if shifts[action] == best_shift:
            pos = pos_of[action]
            pos_counts[pos] = pos_counts.get(pos, 0) + records[j].second
    best_count = 0
    for value, count in pos_counts.items():
        pos = value
        if count > best_count or (count == best_count and pos_names[pos] < pos_names[best_pos]):
            best_count = count
            best_pos = pos
    best_count = 0
    for j in range(start, end):
        action = <uint32_t>records[j].first
        count = records[j].second
        if shifts[action] == best_shift and pos_of[action] == best_pos:
            if count > best_count or (count == best_count and actions[action][1:] < actions[best_action][1:]):
                best_count = count
                best_action = action
    return best_action


cdef int32_t _fallback(
        uint32_t node, uint32_t context,
        vector[uint32_t]& parents, vector[int32_t]& defaults,
        unordered_map[uint64_t, uint32_t]& contexts):
    cdef unordered_map[uint64_t, uint32_t].iterator it
    while node:
        it = contexts.find((<uint64_t>node << 32) | context)
        if it != contexts.end():
            return <int32_t>deref(it).second
        if defaults[node] >= 0:
            return defaults[node]
        node = parents[node]
    return -1


def extract_patterns(sentences, dictionary=(), max_pattern_chars=None,
                     bint prune=True, int min_count=1):
    """Extract a deterministic, prefix-pruned feature-sequence model.

    Each returned record is ``(prefix, previous_pos, shift, pos, lemma_strip,
    lemma_suffix, reading)``.  All lengths count Unicode code points, and
    ``previous_pos == ''`` is beginning of sentence. ``None`` means no POS
    condition.  The default context limit is the longest dictionary or corpus
    word. An explicitly lower limit still retains each complete corpus word.

    ``min_count`` applies to the total count of an observed candidate, not to
    individual output labels. It never removes an unobserved dictionary seed.
    """
    cdef _SurfaceTrie trie = _SurfaceTrie()
    cdef dict pos_ids = {"": 0}, action_ids = {}, dictionary_seeds = {}
    cdef list pos_names = [""], actions = [], patterns = []
    cdef vector[uint32_t] shifts, pos_of
    cdef vector[uint64_t] pos_counts
    cdef unordered_map[uint64_t, uint32_t] candidates
    cdef vector[uint64_t] candidate_keys
    cdef unordered_map[uint64_t, uint64_t] counts
    cdef unordered_map[uint64_t, uint64_t].iterator count_it
    cdef unordered_map[uint64_t, uint32_t].iterator candidate_it
    cdef vector[pair[uint64_t, uint64_t]] records
    cdef vector[int32_t] winners, generic, kept_generic, uniform, first_context, next_context
    cdef vector[uint64_t] support
    cdef unordered_map[uint64_t, uint32_t] kept_contexts
    cdef vector[uint32_t] kept_candidate_ids
    cdef uint32_t node, parent, action, chosen, previous, candidate_id, pos, context
    cdef uint64_t key, total, observation_updates = 0
    cdef size_t j, start, end, candidate_count, count_count, node_count
    cdef Py_ssize_t offset, k, word_length, cap, sentence_length, max_word = 0, max_dict = 0
    cdef Py_ssize_t token_count = 0, sentence_count = 0, dictionary_count = 0
    cdef int32_t context_id, inherited, best_prior, selected
    cdef uint64_t prior, prior_best
    cdef tuple token, out
    cdef object sentence, row, alternatives, item
    cdef unicode text, prefix
    cdef list checked, local_patterns

    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if max_pattern_chars is not None:
        if isinstance(max_pattern_chars, bool) or not isinstance(max_pattern_chars, int) or max_pattern_chars < 1:
            raise ValueError("max_pattern_chars must be a positive integer or None")
        if max_pattern_chars > 0x7fffffff:
            raise ValueError("max_pattern_chars is too large")
    # A one-shot sentence iterable is supported without making copies of tokens.
    if not isinstance(sentences, (list, tuple)):
        sentences = list(sentences)
    for sentence in sentences:
        for row in sentence:
            token = _check_token(row)
            max_word = max(max_word, len(token[0]))

    for row in dictionary:
        token = _check_token(row)
        dictionary_count += 1
        max_dict = max(max_dict, len(token[0]))
        node = trie.insert(token[0])
        action = _intern_action(token, pos_ids, pos_names, action_ids, actions, shifts, pos_of)
        alternatives = dictionary_seeds.get(node)
        if alternatives is None:
            dictionary_seeds[node] = [action]
        elif action not in alternatives:
            alternatives.append(action)

    cap = max_pattern_chars if max_pattern_chars is not None else max(max_dict, max_word, 1)
    for sentence in sentences:
        if not sentence:
            continue
        checked = [_check_token(row) for row in sentence]
        text = "".join(row[0] for row in checked)
        sentence_length = len(text)
        sentence_count += 1
        offset = 0
        previous = 0
        for token in checked:
            token_count += 1
            word_length = len(token[0])
            action = _intern_action(token, pos_ids, pos_names, action_ids, actions, shifts, pos_of)
            pos = pos_of[action]
            if pos_counts.size() <= pos:
                pos_counts.resize(pos + 1, 0)
            pos_counts[pos] += 1
            node = 0
            # Full enumeration: no occurrence-order-dependent early stopping.
            for k in range(min(sentence_length - offset, max(cap, word_length))):
                node = trie.extend(node, PyUnicode_READ_CHAR(text, offset + k))
                if k + 1 < word_length:
                    continue
                candidate_id = _candidate(node, 0, candidates, candidate_keys)
                key = (<uint64_t>candidate_id << 32) | action
                counts[key] += 1
                candidate_id = _candidate(node, previous + 1, candidates, candidate_keys)
                key = (<uint64_t>candidate_id << 32) | action
                counts[key] += 1
                observation_updates += 2
            offset += word_length
            previous = pos
    if pos_counts.size() < <size_t>len(pos_names):
        pos_counts.resize(len(pos_names), 0)

    candidate_count = candidate_keys.size()
    node_count = trie.parent.size()
    count_count = counts.size()
    records.reserve(count_count)
    count_it = counts.begin()
    while count_it != counts.end():
        records.push_back(pair[uint64_t, uint64_t](deref(count_it).first, deref(count_it).second))
        inc(count_it)
    counts.clear()
    counts.rehash(0)
    sort(records.begin(), records.end())
    winners.resize(candidate_count, -1)
    support.resize(candidate_count, 0)
    start = 0
    while start < records.size():
        candidate_id = <uint32_t>(records[start].first >> 32)
        end = start + 1
        while end < records.size() and <uint32_t>(records[end].first >> 32) == candidate_id:
            end += 1
        chosen = _choose_action(records, start, end, shifts, pos_of, actions, pos_names, &total)
        support[candidate_id] = total
        if total >= <uint64_t>min_count:
            winners[candidate_id] = chosen
        start = end
    records.clear()
    records.shrink_to_fit()

    generic.resize(node_count, -1)
    kept_generic.resize(node_count, -1)
    uniform.resize(node_count, -1)
    first_context.resize(node_count, -1)
    next_context.resize(candidate_count, -1)
    for j in range(candidate_count):
        if winners[j] < 0:
            continue
        key = candidate_keys[j]
        node = <uint32_t>(key >> 32)
        context = <uint32_t>key
        if context == 0:
            generic[node] = winners[j]
        else:
            next_context[j] = first_context[node]
            first_context[node] = j

    # Dictionary entries are priors only when no retained corpus pattern exists
    # for their exact surface.  Corpus costs and Viterbi transitions are unused.
    for item, alternatives in dictionary_seeds.items():
        node = item
        if generic[node] >= 0:
            continue
        selected = -1
        prior_best = 0
        for row in alternatives:
            action = row
            prior = pos_counts[pos_of[action]]
            if selected < 0 or prior > prior_best or (prior == prior_best and actions[action] < actions[selected]):
                selected = action
                prior_best = prior
        generic[node] = selected
    dictionary_seeds.clear()
    candidates.clear()
    candidates.rehash(0)
    trie.edges.clear()
    trie.edges.rehash(0)
    action_ids.clear()

    # Parents always precede children.  A generic pattern is removed only when
    # the inherited output is provably constant for *every* previous POS.
    # This conservatively preserves context resets that a naive prune loses.
    for node in range(1, node_count):
        parent = trie.parent[node]
        selected = generic[node]
        if selected >= 0 and (not prune or uniform[parent] != selected):
            kept_generic[node] = selected
        inherited = kept_generic[node] if kept_generic[node] >= 0 else uniform[parent]
        uniform[node] = inherited
        context_id = first_context[node]
        while context_id >= 0:
            context = <uint32_t>candidate_keys[context_id]
            selected = winners[context_id]
            if kept_generic[node] >= 0:
                best_prior = kept_generic[node]
            else:
                best_prior = _fallback(parent, context, trie.parent, kept_generic, kept_contexts)
            if not prune or selected != best_prior:
                kept_contexts[(<uint64_t>node << 32) | context] = selected
                kept_candidate_ids.push_back(context_id)
                if selected != inherited:
                    uniform[node] = -1
            context_id = next_context[context_id]

    for node in range(1, node_count):
        selected = kept_generic[node]
        if selected >= 0:
            prefix = trie.surface(node)
            out = actions[selected]
            patterns.append((prefix, None, shifts[selected], out[0], out[1], out[2], out[3]))
    for j in range(kept_candidate_ids.size()):
        candidate_id = kept_candidate_ids[j]
        key = candidate_keys[candidate_id]
        node = <uint32_t>(key >> 32)
        context = <uint32_t>key
        selected = winners[candidate_id]
        prefix = trie.surface(node)
        out = actions[selected]
        patterns.append((prefix, pos_names[context - 1], shifts[selected], out[0], out[1], out[2], out[3]))
    patterns.sort(key=lambda p: (p[0], p[1] is not None, p[1] or ""))
    stats = {
        "algorithm": "Yoshinaga-2023-feature-sequence-trie",
        "implementation": "independent Cython extractor; exhaustive right context",
        "sentences": sentence_count,
        "tokens": token_count,
        "dictionary_entries": dictionary_count,
        "dictionary_max_word_chars": max_dict,
        "corpus_max_word_chars": max_word,
        "max_pattern_chars": cap,
        "candidate_patterns": candidate_count,
        "count_records": count_count,
        "observation_updates": observation_updates,
        "training_trie_nodes": node_count,
        "patterns": len(patterns),
        "context_patterns": kept_candidate_ids.size(),
        "pos_labels": len(pos_names) - 1,
        "morphological_labels": len(actions),
        "prune": bool(prune),
        "min_count": min_count,
        "tie_breaking": "longer shift; lexicographic POS and lemma/read metadata",
        "length_unit": "Unicode code points",
    }
    return patterns, stats
