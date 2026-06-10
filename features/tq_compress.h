#ifndef GREENBOOST_TQ_COMPRESS_H
#define GREENBOOST_TQ_COMPRESS_H

#include <stdint.h>

/* Per-allocation compression metadata stored alongside the hash-map entry.
 * When GB_KV_COMPRESS_FLAG is set in an entry's flags, the T2 copy holds
 * absmax int8 data (half the size of fp16) plus per-row scale factors. */

#define GB_KV_COMPRESS_ROW   128   /* elements per absmax quantisation row  */
#define GB_KV_COMPRESS_FLAG  0x80  /* OR into gb_buf flags to mark compressed */

struct gb_kv_compressed {
    uint8_t  *int8_data;   /* absmax-quantised int8 copy in T2 (n_elems bytes)       */
    float    *scales;      /* per-row scale factors: 1 float per GB_KV_COMPRESS_ROW  */
    uint64_t  orig_size;   /* original fp16/bf16 byte count                          */
    uint32_t  n_rows;      /* number of GB_KV_COMPRESS_ROW-element quantisation rows  */
    uint8_t   compressed;  /* 1 if this entry holds compressed data                  */
    uint8_t   orig_dtype;  /* 0 = fp16, 1 = bf16                                     */
};

#endif /* GREENBOOST_TQ_COMPRESS_H */
