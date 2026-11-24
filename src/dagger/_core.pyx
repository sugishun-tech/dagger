# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Original static double-array compiler and UTF-8 inference kernel."""
from libc.stdint cimport uint32_t, int32_t, uint8_t, uint64_t
from libc.stdlib cimport calloc, free
from libc.string cimport memchr
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE, PyBytes_FromStringAndSize
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
cimport cython


cdef struct Node:
    int32_t base
    uint32_t check
    int32_t value
    uint32_t flags


cdef struct BuildNode:
    uint32_t parent
    uint32_t code
    uint32_t degree
    uint32_t first_child
    uint32_t state
    int32_t base
    int32_t value
    uint32_t flags


cdef struct Label:
    uint32_t shift
    uint32_t byte_shift
    uint32_t pos
    uint32_t strip
    uint32_t byte_strip
    uint32_t kind


cdef struct Hit:
    size_t start
    size_t end
    uint32_t label
    uint8_t unknown


cdef void _grow_build_slots(vector[Node]& nodes, size_t needed,
                           size_t* first_free, size_t* last_free) except *:
    """Unused base/value fields form a doubly linked, ordered free-slot list."""
    cdef size_t old_size = nodes.size(), new_size, i
    if needed <= old_size:
        return
    if needed >= 0x80000000:
        raise OverflowError("Model exceeds 32-bit trie capacity")
    new_size = max(needed + 4096, old_size * 2)
    if new_size >= 0x80000000:
        new_size = needed
    nodes.resize(new_size)
    if last_free[0]:
        nodes[last_free[0]].value = <int32_t>old_size
    else:
        first_free[0] = old_size
    for i in range(old_size, new_size):
        nodes[i].base = <int32_t>(last_free[0] if i == old_size else i - 1)
        nodes[i].value = <int32_t>(i + 1 if i + 1 < new_size else 0)
    last_free[0] = new_size - 1


cdef inline void _take_build_slot(vector[Node]& nodes, uint32_t target,
                                  uint32_t owner, size_t* first_free,
                                  size_t* last_free) noexcept:
    cdef uint32_t before = <uint32_t>nodes[target].base
    cdef uint32_t after = <uint32_t>nodes[target].value
    if before:
        nodes[before].value = after
    else:
        first_free[0] = after
    if after:
        nodes[after].base = before
    else:
        last_free[0] = before
    nodes[target].check = owner + 1


def build_double_array(entries, unsigned int character_count):
    """Compile sorted unique keys into a densely packed static double array.

    Logical trie nodes are built from successive longest common prefixes.
    Sibling groups are then placed by descending degree, independently of their
    parents' physical addresses. Unary groups fill the remaining holes. The
    final pass resolves logical parents to physical ``check`` addresses.
    """
    cdef vector[uint32_t] key, previous_key, path, children, cursors
    cdef vector[BuildNode] logical
    cdef vector[pair[uint32_t, uint32_t]] order
    cdef vector[Node] nodes
    cdef BuildNode child
    cdef size_t i, j, k, common, shared, group_start, group_end, total_children = 0
    cdef size_t first_free = 0, last_free = 0, anchor, needed, maximum = 1
    cdef size_t search_cursor = 2
    cdef size_t attempts, candidate
    cdef long long offset
    cdef uint32_t parent, owner, code, minimum_code, maximum_code, state, target
    cdef uint32_t degree, previous_degree = 0
    cdef bint fits, have_key = False
    cdef object item, symbol, value
    if character_count >= 0x40000000:
        raise ValueError("Character alphabet exceeds trie symbol range")
    child.parent = 0
    child.code = 0
    child.degree = 0
    child.first_child = 0
    child.state = 1
    child.base = 0
    child.value = -1
    child.flags = 0
    logical.push_back(child)
    path.push_back(0)
    for item in entries:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each trie entry must be a (key, value) pair")
        value = item[1]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 0x80000000:
            raise ValueError("Trie value must be a nonnegative 32-bit integer")
        key.clear()
        for symbol in item[0]:
            if isinstance(symbol, bool) or not isinstance(symbol, int) or not 1 <= symbol < 0x40000000:
                raise ValueError("Trie symbol out of range")
            key.push_back(symbol)
        if key.empty():
            raise ValueError("Empty trie key")
        common = 0
        shared = min(key.size(), previous_key.size())
        while common < shared and key[common] == previous_key[common]:
            common += 1
        if have_key:
            if common == key.size():
                if key.size() == previous_key.size():
                    raise ValueError("Duplicate trie key")
                raise ValueError("Trie keys must be sorted in increasing order")
            if common < previous_key.size() and key[common] < previous_key[common]:
                raise ValueError("Trie keys must be sorted in increasing order")
        path.resize(common + 1)
        for j in range(common, key.size()):
            if logical.size() >= 0x7fffffff:
                raise OverflowError("Model exceeds 32-bit logical trie capacity")
            parent = path.back()
            code = key[j]
            logical[parent].degree += 1
            logical[parent].flags |= 1 if code <= character_count else 2
            child.parent = parent
            child.code = code
            child.degree = 0
            child.first_child = 0
            child.state = 0
            child.base = 0
            child.value = -1
            child.flags = 0
            path.push_back(<uint32_t>logical.size())
            logical.push_back(child)
        logical[path.back()].value = value
        previous_key.swap(key)
        have_key = True
    if not have_key:
        raise ValueError("Empty trie")

    # CSR children stay in symbol order because keys were strictly sorted.
    cursors.resize(logical.size())
    children.resize(logical.size() - 1)
    for i in range(logical.size()):
        logical[i].first_child = total_children
        total_children += logical[i].degree
        if logical[i].degree:
            order.push_back(pair[uint32_t, uint32_t](0xffffffff - logical[i].degree, i))
    for i in range(1, logical.size()):
        parent = logical[i].parent
        children[logical[parent].first_child + cursors[parent]] = i
        cursors[parent] += 1
    cursors.clear()
    cursors.shrink_to_fit()
    sort(order.begin(), order.end())

    nodes.resize(2)
    nodes[1].check = 1
    nodes[1].value = -1
    _grow_build_slots(nodes, 4096, &first_free, &last_free)
    for i in range(order.size()):
        owner = order[i].second
        degree = logical[owner].degree
        group_start = logical[owner].first_child
        group_end = group_start + degree
        minimum_code = logical[children[group_start]].code
        maximum_code = logical[children[group_end - 1]].code
        if not first_free:
            _grow_build_slots(nodes, nodes.size() + 1, &first_free, &last_free)
        if degree == 1 or degree != previous_degree:
            anchor = first_free
        else:
            anchor = max(first_free, search_cursor)
            _grow_build_slots(nodes, anchor + 1, &first_free, &last_free)
            while nodes[anchor].check:
                anchor += 1
                _grow_build_slots(nodes, anchor + 1, &first_free, &last_free)
        previous_degree = degree
        attempts = 0
        while True:
            needed = anchor + maximum_code - minimum_code + 1
            _grow_build_slots(nodes, needed, &first_free, &last_free)
            offset = <long long>anchor - minimum_code
            fits = True
            # The first child maps to the already-free anchor.
            for k in range(group_start + 1, group_end):
                candidate = <size_t>(offset + logical[children[k]].code)
                if nodes[candidate].check:
                    fits = False
                    break
            if fits:
                break
            attempts += 1
            if attempts >= 2048:
                # Beyond the current high-water mark all targets are free.
                anchor = maximum + 1
            else:
                anchor = <uint32_t>nodes[anchor].value
                if not anchor:
                    anchor = maximum + 1
        logical[owner].base = <int32_t>offset
        # Continue past the last successful anchor for the same degree. This
        # avoids retrying the same unsuitable early holes for every group.
        # A lower degree starts again at the earliest hole; unary nodes pack
        # every remaining hole without any cursor restriction.
        search_cursor = anchor + 1
        for k in range(group_start, group_end):
            parent = children[k]
            target = <uint32_t>(offset + logical[parent].code)
            _take_build_slot(nodes, target, parent, &first_free, &last_free)
            logical[parent].state = target
            maximum = max(maximum, <size_t>target)

    # Clear free-list bookkeeping before serializing the portable node array.
    for i in range(2, maximum + 1):
        if not nodes[i].check:
            nodes[i].base = 0
            nodes[i].value = 0
    for i in range(logical.size()):
        state = logical[i].state
        if not state:
            raise RuntimeError("Unassigned logical trie state")
        nodes[state].base = logical[i].base
        nodes[state].check = 1 if i == 0 else logical[logical[i].parent].state
        nodes[state].value = logical[i].value
        nodes[state].flags = logical[i].flags
    return PyBytes_FromStringAndSize(<const char*>&nodes[0], (maximum + 1) * sizeof(Node))


cdef inline uint32_t decode(const unsigned char* data, size_t* offset) noexcept nogil:
    """Input is validated as UTF-8 at the public boundary."""
    cdef size_t i = offset[0]
    cdef uint32_t first = data[i], result
    if first < 0x80:
        offset[0] = i + 1
        return first
    if first < 0xE0:
        result = ((first & 31) << 6) | (data[i + 1] & 63)
        offset[0] = i + 2
    elif first < 0xF0:
        result = ((first & 15) << 12) | ((data[i + 1] & 63) << 6) | (data[i + 2] & 63)
        offset[0] = i + 3
    else:
        result = ((first & 7) << 18) | ((data[i + 1] & 63) << 12) | ((data[i + 2] & 63) << 6) | (data[i + 3] & 63)
        offset[0] = i + 4
    return result


cdef inline int character_kind(uint32_t c) noexcept nogil:
    if 48 <= c <= 57 or 0xFF10 <= c <= 0xFF19:
        return 1
    if (c == 0x3007 or c == 0x4E00 or c == 0x4E8C or c == 0x4E09 or
        c == 0x56DB or c == 0x4E94 or c == 0x516D or c == 0x4E03 or
        c == 0x516B or c == 0x4E5D or c == 0x5341 or c == 0x767E or
        c == 0x5343 or c == 0x4E07 or c == 0x5104 or c == 0x5146 or
        c == 0x6570 or c == 0x30FB):
        return 1
    if 65 <= c <= 90 or 97 <= c <= 122 or 0xFF21 <= c <= 0xFF3A or 0xFF41 <= c <= 0xFF5A:
        return 2
    if (0x30A1 <= c <= 0x30FA) or (0x30FC <= c <= 0x30FF) or (0xFF66 <= c <= 0xFF9F):
        return 3
    if c <= 32 or c == 0xA0 or c == 0x3000 or (0x2000 <= c <= 0x200A):
        return 4
    if c < 128 or 0x2000 <= c <= 0x206F or 0x3000 <= c <= 0x303F or 0x1F000 <= c <= 0x1FAFF:
        return 5
    return 0


def classify_word(bytes data):
    """Compute a reusable character-kind feature during model compilation."""
    cdef const unsigned char* raw = <const unsigned char*>PyBytes_AS_STRING(data)
    cdef size_t n = PyBytes_GET_SIZE(data), at = 0
    cdef int kind
    data.decode('utf-8')
    if not n:
        return 0
    kind = character_kind(decode(raw, &at))
    while at < n:
        if character_kind(decode(raw, &at)) != kind:
            return 0
    return kind


@cython.final
cdef class Engine:
    cdef const int32_t[::1] storage
    cdef const Node* nodes
    cdef size_t node_count
    cdef uint32_t* cmap
    cdef uint32_t context_offset
    cdef vector[Label] labels
    cdef vector[string] prefixes
    cdef vector[string] suffixes
    cdef vector[string] readings
    cdef vector[uint32_t] unknown_labels
    cdef public object manifest

    def __cinit__(self):
        self.cmap = NULL

    def __init__(self, const int32_t[::1] storage, manifest):
        cdef size_t i, j, current, parent, limit
        cdef long long symbol
        cdef Label label
        cdef vector[uint32_t] depths
        cdef vector[uint32_t] character_depths
        cdef vector[uint32_t] chain
        cdef vector[uint8_t] marks
        cdef vector[uint8_t] flags
        cdef vector[uint8_t] widths
        cdef uint32_t char_count, byte_depth, edge
        if storage.shape[0] < 8 or storage.shape[0] % 4:
            raise ValueError("Invalid trie buffer")
        from .model import _validate_manifest
        _validate_manifest(manifest, storage.shape[0] * sizeof(int32_t))
        self.storage = storage
        self.nodes = <const Node*>&self.storage[0]
        self.node_count = storage.shape[0] // 4
        self.manifest = manifest
        self.context_offset = manifest['context_offset']
        char_count = len(manifest['characters'])
        limit = char_count + len(manifest['pos'])
        self.cmap = <uint32_t*>calloc(0x110000, sizeof(uint32_t))
        if self.cmap == NULL:
            raise MemoryError()
        widths.resize(limit + 1)
        for i, c in enumerate(manifest['characters']):
            self.cmap[c] = i + 1
            widths[i + 1] = len(chr(c).encode('utf-8'))
        for pos in manifest['pos']:
            self.prefixes.push_back((pos + ',').encode('utf-8'))
        for row in manifest['labels']:
            label.shift = row[0]
            label.byte_shift = row[1]
            label.pos = row[2]
            label.strip = row[3]
            label.byte_strip = row[4]
            label.kind = row[7]
            self.labels.push_back(label)
            self.suffixes.push_back(row[5].encode('utf-8'))
            self.readings.push_back((',' + row[6] + ',*\n').encode('utf-8'))
        for i in manifest['unknown_labels']:
            self.unknown_labels.push_back(i)
        # Validate parent chains, transition alphabets, flags and label bounds.
        # This runs once on model load, never in the input hot loop.
        depths.resize(self.node_count)
        character_depths.resize(self.node_count)
        marks.resize(self.node_count)
        flags.resize(self.node_count)
        if self.nodes[1].check != 1 or self.nodes[1].value != -1:
            raise ValueError("Invalid trie root")
        marks[1] = 2
        for i in range(2, self.node_count):
            if not self.nodes[i].check:
                continue
            parent = self.nodes[i].check
            if parent >= self.node_count or parent == i or not self.nodes[parent].check:
                raise ValueError("Invalid trie parent")
            symbol = <long long>i - self.nodes[parent].base
            if symbol <= 0 or symbol > limit:
                raise ValueError("Invalid trie transition")
            flags[parent] |= 1 if symbol <= char_count else 2
            if self.nodes[i].value < -1 or <long long>self.nodes[i].value >= <long long>self.labels.size():
                raise ValueError("Invalid trie label")
            chain.clear()
            current = i
            while marks[current] != 2:
                if marks[current] == 1:
                    raise ValueError("Cycle in trie")
                marks[current] = 1
                chain.push_back(current)
                parent = self.nodes[current].check
                if parent == 0 or parent >= self.node_count:
                    raise ValueError("Invalid trie ancestor")
                current = parent
            while not chain.empty():
                current = chain.back()
                chain.pop_back()
                parent = self.nodes[current].check
                symbol = <long long>current - self.nodes[parent].base
                if symbol <= 0 or symbol > limit:
                    raise ValueError("Invalid trie alphabet")
                depths[current] = depths[parent] + widths[symbol]
                character_depths[current] = character_depths[parent] + (1 if symbol <= char_count else 0)
                marks[current] = 2
        for i in range(1, self.node_count):
            if not self.nodes[i].check:
                continue
            if self.nodes[i].flags != flags[i]:
                raise ValueError("Invalid trie flags")
            if i > 1:
                parent = self.nodes[i].check
                symbol = <long long>i - self.nodes[parent].base
                if symbol > char_count and self.nodes[i].flags:
                    raise ValueError("A context transition must be terminal")
            if self.nodes[i].value >= 0:
                label = self.labels[self.nodes[i].value]
                if label.byte_shift == 0 or label.byte_shift > depths[i]:
                    raise ValueError("Invalid pattern shift")
                current = i
                while character_depths[current] > label.shift:
                    current = self.nodes[current].check
                if character_depths[current] != label.shift or depths[current] != label.byte_shift:
                    raise ValueError("Pattern shift is not a UTF-8 boundary")
                while character_depths[current] > label.shift - label.strip:
                    current = self.nodes[current].check
                if depths[current] != label.byte_shift - label.byte_strip:
                    raise ValueError("Lemma strip is not a UTF-8 boundary")

    def __dealloc__(self):
        free(self.cmap)

    cdef inline int lookup(self, const unsigned char* data, size_t start, size_t end,
                           uint32_t previous) noexcept nogil:
        cdef uint32_t state = 1, code, cp
        cdef long long target
        cdef size_t at = start
        while at < end and self.nodes[state].flags & 1:
            cp = decode(data, &at)
            code = self.cmap[cp]
            if code == 0:
                break
            target = <long long>self.nodes[state].base + code
            if target < 2 or target >= self.node_count or self.nodes[target].check != state:
                break
            state = <uint32_t>target
        # Surface length wins before previous-POS specificity.
        while state != 1:
            if self.nodes[state].flags & 2:
                target = <long long>self.nodes[state].base + self.context_offset + previous
                if 2 <= target < self.node_count and self.nodes[target].check == state:
                    return self.nodes[target].value
            if self.nodes[state].value >= 0:
                return self.nodes[state].value
            state = self.nodes[state].check
        return -1

    cdef size_t scan(self, const unsigned char* data, size_t start, size_t end,
                     vector[Hit]& hits, bint collect) except? -1 nogil:
        cdef size_t at = start, after, q, last_unit = 0, unit, count = 0
        cdef uint32_t previous = 0, cp
        cdef int index, kind, last_kind = 0
        cdef bint unknown, merge
        cdef Hit hit
        hits.clear()
        while at < end:
            index = self.lookup(data, at, end, previous)
            unknown = index < 0
            if unknown:
                q = at
                cp = decode(data, &q)
                kind = character_kind(cp)
                after = q
                if kind == 1:
                    index = self.unknown_labels[1]
                elif kind == 4:
                    index = self.unknown_labels[3]
                elif kind == 5:
                    index = self.unknown_labels[2]
                else:
                    index = self.unknown_labels[0]
            else:
                after = at + self.labels[index].byte_shift
                kind = self.labels[index].kind
                # Defensive check also protects custom model containers.
                if after > end or after <= at:
                    q = at
                    decode(data, &q)
                    after = q
                    index = self.unknown_labels[0]
                    unknown = True
                    kind = 0
            unit = after - at
            merge = (count > 0 and kind != 0 and kind <= 4 and last_kind == kind)
            if kind == 3 and last_unit + unit >= 18:
                merge = False
            if merge:
                if collect:
                    hits.back().end = after
                    hits.back().label = index
                    hits.back().unknown = 1
            else:
                count += 1
                if collect:
                    hit.start = at
                    hit.end = after
                    hit.label = index
                    hit.unknown = unknown
                    hits.push_back(hit)
            previous = self.labels[index].pos
            last_kind = kind
            last_unit = unit
            at = after
        return count

    def spans(self, bytes data):
        cdef const unsigned char* raw = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef size_t size = PyBytes_GET_SIZE(data), i
        cdef vector[Hit] hits
        data.decode('utf-8')
        with nogil:
            self.scan(raw, 0, size, hits, True)
        return [(hits[i].start, hits[i].end, hits[i].label, bool(hits[i].unknown))
                for i in range(hits.size())]

    cdef int render_sentence(self, const unsigned char* raw, size_t start, size_t end,
                              vector[Hit]& hits, string& output, bint wakati) except -1 nogil:
        cdef size_t i, length
        cdef uint32_t index
        self.scan(raw, start, end, hits, True)
        for i in range(hits.size()):
            index = hits[i].label
            length = hits[i].end - hits[i].start
            output.append(<const char*>(raw + hits[i].start), length)
            if wakati:
                output.push_back(32)
            else:
                output.push_back(9)
                output.append(self.prefixes[self.labels[index].pos])
                if hits[i].unknown:
                    output.append(b'*,*,*\n')
                else:
                    output.append(<const char*>(raw + hits[i].start), length - self.labels[index].byte_strip)
                    output.append(self.suffixes[index])
                    output.append(self.readings[index])
        if wakati:
            output.push_back(10)
        else:
            output.append(b'EOS\n')
        return 0

    def parse(self, bytes data, bint wakati=False, bint batch=False):
        cdef const unsigned char* raw = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef const char* newline
        cdef size_t size = PyBytes_GET_SIZE(data), at = 0, end, next_at
        cdef vector[Hit] hits
        cdef string output
        data.decode('utf-8')
        with nogil:
            output.reserve(size * (2 if wakati else 5))
            if not batch:
                self.render_sentence(raw, 0, size, hits, output, wakati)
            else:
                while at < size:
                    newline = <const char*>memchr(raw + at, 10, size - at)
                    if newline == NULL:
                        end = size
                        next_at = size
                    else:
                        end = newline - <const char*>raw
                        next_at = end + 1
                    if end > at and raw[end - 1] == 13:
                        end -= 1
                    self.render_sentence(raw, at, end, hits, output, wakati)
                    at = next_at
        return PyBytes_FromStringAndSize(output.data(), output.size())

    def count(self, bytes data, bint batch=False):
        cdef const unsigned char* raw = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef const char* newline
        cdef size_t size = PyBytes_GET_SIZE(data), at = 0, end, next_at, result = 0
        cdef vector[Hit] hits
        data.decode('utf-8')
        with nogil:
            if not batch:
                result = self.scan(raw, 0, size, hits, False)
            else:
                while at < size:
                    newline = <const char*>memchr(raw + at, 10, size - at)
                    if newline == NULL:
                        end = size
                        next_at = size
                    else:
                        end = newline - <const char*>raw
                        next_at = end + 1
                    if end > at and raw[end - 1] == 13:
                        end -= 1
                    result += self.scan(raw, at, end, hits, False)
                    at = next_at
        return result

