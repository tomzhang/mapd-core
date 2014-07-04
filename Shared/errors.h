/* 
 * File:        errors.h
 * Author(s):   steve@map-d.com
 *
 * Created on June 19, 2014, 4:22 PM
 */

#ifndef ERRORS_H
#define ERRORS_H

typedef enum _mapd_err_t {
    MAPD_FAILURE            	= 0,    // generic error number
    MAPD_SUCCESS            	= 1,    // success! Map-Delicious!
    MAPD_ERR_FILE_OPEN      	= -1,   // unable to open file
    MAPD_ERR_FILE_CLOSE     	= -2,   // error closing file
    MAPD_ERR_FILE_WRITE     	= -3,   // error writing file
    MAPD_ERR_FILE_READ      	= -4,   // error reading file
    MAPD_ERR_FILE_APPEND    	= -5,   // error appending to file
    MAPD_ERR_FILE_NOT_FOUND 	= -6,	// file not found
    MAPD_ERR_CHUNK_NOT_FOUND	= -7,	// chunk not found
    MAPD_ERR_CHUNK_DUPL			= -8,   // chunk duplicate
    MAPD_ERR_BUFFER         	= -9    // error, buffer manager
} mapd_err_t;

#define PERROR(errNum, func, lineNo, msg) (fprintf(stderr, "[%s:%u] ERROR(%d): %s\n", func, lineNo, errNum, msg))

#endif	/* ERRORS_H */

