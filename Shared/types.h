/*
 * File:        types.h
 * Author(s):   steve@map-d.com
 *
 * Created on June 19, 2014, 4:29 PM
 */

#ifndef _TYPES_H
#define _TYPES_H

#include <vector>
#include <cstddef>

// The ChunkKey is a unique identifier for chunks in the database file.
// The first element of the underlying vector for ChunkKey indicates the type of
// ChunkKey (also referred to as the keyspace id)
typedef std::vector<int> ChunkKey;

#endif /* _TYPES_H */
