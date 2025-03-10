/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers
 * University
 *
 * See COPYRIGHT in top-level directory.
 */

#ifndef __SS_DATA_H_
#define __SS_DATA_H_

#include <stdlib.h>

#include "bbox.h"
#include "list.h"
#include "str_hash.h"
#include <margo.h>
#include <mercury.h>
#include <mercury_atomic.h>
#include <mercury_bulk.h>
#include <mercury_macros.h>
#include <mercury_proc_string.h>

#include "dspaces-common.h"

#include <abt.h>

#define MAX_VERSIONS 10

#define DS_CLIENT_STORAGE 0x01
#define DS_OBJ_RESIZE 0x02
#define DS_NO_COMPRESS 0x04

typedef struct {
    void *iov_base;
    size_t iov_len;
} iovec_t;

enum storage_type { row_major, column_major };

#define OD_MAX_NAME_LEN 150
#define OD_MAX_OWNER_LEN 128

typedef struct {
    char name[OD_MAX_NAME_LEN];

    enum storage_type st;
    uint32_t flags;
    int tag;
    int type;

    char owner[OD_MAX_OWNER_LEN];
    unsigned int version;

    /* Global bounding box descriptor. */
    struct bbox bb;

    /* Size of one element of a data object. */
    size_t size;
} obj_descriptor;

struct meta_data {
    struct list_head entry;
    char *name;
    unsigned int version;
    int length;
    void *data;
};

struct global_dimension {
    int ndim;
    struct coord sizes;
};

struct obj_data {
    struct list_head obj_entry;

    obj_descriptor obj_desc;

    struct global_dimension gdim;

    void *data; /* Aligned pointer */

    /* Reference to the parent object; used only for sub-objects. */
    struct obj_data *obj_ref;

    /* Count how many references are to this data object. */
    int refcnt;

    /* Flag to mark if we should free this data object. */
    unsigned int f_free : 1;
};

struct gdim_list_entry {
    struct list_head entry;
    char *var_name;
    struct global_dimension gdim;
};

struct meta_sub_list_entry {
    int version;
    const char *name;
    struct meta_data *mdata;
    struct list_head entry;
};

typedef struct {
    int num_obj;
    int size_hash;
    ds_str_hash *var_dict;
    struct list_head *meta_hash;
    ABT_mutex *meta_mutex;
    ABT_cond *meta_cond;
    struct list_head *meta_subs;

    /* List of data objects. */
    struct list_head obj_hash[1];
} ss_storage;

struct obj_desc_list {
    struct list_head odsc_entry;
    obj_descriptor odsc;
};

struct obj_desc_ptr_list {
    struct list_head odsc_entry;
    obj_descriptor *odsc;
};

typedef struct {
    size_t size;
    char *raw_odsc;
} odsc_hdr;

typedef struct {
    size_t size;
    size_t gdim_size;
    char *raw_odsc;
    char *raw_gdim;

} odsc_hdr_with_gdim;

struct dht_sub_list_entry {
    obj_descriptor *odsc; // subbed object
    long remaining;
    int pub_count;
    struct list_head recv_odsc;
    struct list_head entry;
};

struct dht_entry {
    /* Global info. */
    struct sspace *ss;
    struct bbox bb;

    int rank;

    struct intv i_virt;

    int num_intv;
    struct intv *i_tab;

    int num_bbox;
    int size_bb_tab;
    struct bbox *bb_tab;

    ABT_mutex *hash_mutex;
    ABT_cond *hash_cond;
    struct list_head *dht_subs;

    int odsc_size, odsc_num;
    struct list_head odsc_hash[1];
};

struct dht {
    struct bbox bb_glb_domain;

    int num_entries;
    struct dht_entry *ent_tab[1];
};

enum sspace_hash_version {
    ssd_hash_version_auto = 0,
    ssd_hash_version_v0, // no hashing
    ssd_hash_version_v1, // decompose the global data domain
                         //  using hilbert SFC
    ssd_hash_version_v2, // decompose the global data domain using
                         // recursive bisection of the longest dimension
    _ssd_hash_version_count,
};

/*
  Shared space structure.
*/
struct sspace {
    uint64_t max_dim;
    unsigned int bpd;

    struct dht *dht;

    int rank;
    /* Pointer into "dht.ent_tab" corresponding to this node. */
    struct dht_entry *ent_self;

    // for v2
    int total_num_bbox;
    enum sspace_hash_version hash_version;
};

struct sspace_list_entry {
    struct list_head entry;
    struct global_dimension gdim;
    struct sspace *ssd;
};

typedef struct {
    int num_dims;
    int num_space_srv;
    int max_versions;
    enum sspace_hash_version hash_version;
    struct bbox ss_domain;
    struct global_dimension default_gdim;
    int rank;
} ss_info_hdr;

static inline hg_return_t hg_proc_odsc_hdr(hg_proc_t proc, void *arg)
{
    hg_return_t ret;
    odsc_hdr *in = (odsc_hdr *)arg;
    ret = hg_proc_hg_size_t(proc, &in->size);
    if(ret != HG_SUCCESS)
        return ret;
    if(in->size) {
        switch(hg_proc_get_op(proc)) {
        case HG_ENCODE:
            ret = hg_proc_raw(proc, in->raw_odsc, in->size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_DECODE:
            in->raw_odsc = (char *)malloc(in->size);
            ret = hg_proc_raw(proc, in->raw_odsc, in->size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_FREE:
            free(in->raw_odsc);
            break;
        default:
            break;
        }
    }
    return HG_SUCCESS;
}

static inline hg_return_t hg_proc_odsc_hdr_with_gdim(hg_proc_t proc, void *arg)
{
    hg_return_t ret;
    odsc_hdr_with_gdim *in = (odsc_hdr_with_gdim *)arg;
    ret = hg_proc_hg_size_t(proc, &in->size);
    ret = hg_proc_hg_size_t(proc, &in->gdim_size);
    if(ret != HG_SUCCESS)
        return ret;
    if(in->size) {
        switch(hg_proc_get_op(proc)) {
        case HG_ENCODE:
            ret = hg_proc_raw(proc, in->raw_odsc, in->size);
            if(ret != HG_SUCCESS)
                return ret;
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_DECODE:
            in->raw_odsc = (char *)malloc(in->size);
            ret = hg_proc_raw(proc, in->raw_odsc, in->size);
            if(ret != HG_SUCCESS)
                return ret;
            in->raw_gdim = (char *)malloc(in->gdim_size);
            ret = hg_proc_raw(proc, in->raw_gdim, in->gdim_size);
            if(ret != HG_SUCCESS)
                return ret;
            break;
        case HG_FREE:
            free(in->raw_odsc);
            free(in->raw_gdim);
            break;
        default:
            break;
        }
    }
    return HG_SUCCESS;
}

typedef struct dsp_buf {
    hg_size_t len;
    void *buf;
} dsp_buf_t;

static inline hg_return_t hg_proc_dsp_buf_t(hg_proc_t proc, void *data)
{
    hg_return_t ret;
    dsp_buf_t *buf = (dsp_buf_t *)data;

    switch(hg_proc_get_op(proc)) {
    case HG_ENCODE:
        ret = hg_proc_hg_size_t(proc, &buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        ret = hg_proc_raw(proc, buf->buf, buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        break;
    case HG_DECODE:
        ret = hg_proc_hg_size_t(proc, &buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        buf->buf = malloc(buf->len);
        ret = hg_proc_raw(proc, buf->buf, buf->len);
        if(ret != HG_SUCCESS) {
            break;
        }
        break;
    case HG_FREE:
        free(buf->buf);
        ret = HG_SUCCESS;
    }

    return ret;
}

typedef struct name_list {
    hg_size_t count;
    hg_string_t *names;
} name_list_t;

static inline hg_return_t hg_proc_name_list_t(hg_proc_t proc, void *data)
{
    hg_return_t ret;
    name_list_t *nlist = (name_list_t *)data;
    int i;

    switch(hg_proc_get_op(proc)) {
    case HG_ENCODE:
        ret = hg_proc_hg_size_t(proc, &nlist->count);
        if(ret != HG_SUCCESS) {
            break;
        }
        for(i = 0; i < nlist->count; i++) {
            ret = hg_proc_hg_string_t(proc, &nlist->names[i]);
            if(ret != HG_SUCCESS) {
                break;
            }
        }
        break;
    case HG_DECODE:
        ret = hg_proc_hg_size_t(proc, &nlist->count);
        if(ret != HG_SUCCESS) {
            break;
        }
        nlist->names = malloc(sizeof(*nlist->names) * nlist->count);
        for(i = 0; i < nlist->count; i++) {
            ret = hg_proc_hg_string_t(proc, &nlist->names[i]);
            if(ret != HG_SUCCESS) {
                break;
            }
        }
        break;
    case HG_FREE:
        for(i = 0; i < nlist->count; i++) {
            ret = hg_proc_hg_size_t(proc, &nlist->names[i]);
        }
        free(nlist->names);

        ret = HG_SUCCESS;
        break;
    }

    return (ret);
}

typedef struct dsp_dict {
    hg_size_t len;
    hg_int8_t *types;
    hg_string_t *keys;
    void **vals;
} dsp_dict_t;

static inline hg_return_t dict_encode_val(hg_proc_t proc, hg_int8_t type,
                                          void *val)
{
    switch(type) {
    case DSP_BOOL:
    case DSP_CHAR:
    case DSP_BYTE:
    case DSP_UINT8:
    case DSP_INT8:
        return (hg_proc_raw(proc, val, 1));
    case DSP_UINT16:
    case DSP_INT16:
        return (hg_proc_raw(proc, val, 2));
    case DSP_FLOAT:
    case DSP_INT:
    case DSP_UINT:
    case DSP_UINT32:
    case DSP_INT32:
        return (hg_proc_raw(proc, val, 4));
    case DSP_LONG:
    case DSP_DOUBLE:
    case DSP_ULONG:
    case DSP_UINT64:
    case DSP_INT64:
        return (hg_proc_raw(proc, val, 8));
    case DSP_STR:
    case DSP_JSON:
        return (hg_proc_hg_string_t(proc, &val));
    default:
        return (HG_INVALID_PARAM);
    }
}

static inline hg_return_t dict_decode_val(hg_proc_t proc, hg_int8_t type,
                                          void **val)
{
    hg_return_t ret = HG_SUCCESS;

    switch(type) {
    case DSP_BOOL:
    case DSP_CHAR:
    case DSP_BYTE:
    case DSP_UINT8:
    case DSP_INT8:
        *val = malloc(1);
        return (hg_proc_raw(proc, *val, 1));
    case DSP_UINT16:
    case DSP_INT16:
        *val = malloc(2);
        return (hg_proc_raw(proc, *val, 2));
    case DSP_FLOAT:
    case DSP_INT:
    case DSP_UINT:
    case DSP_UINT32:
    case DSP_INT32:
        *val = malloc(4);
        return (hg_proc_raw(proc, *val, 4));
    case DSP_LONG:
    case DSP_DOUBLE:
    case DSP_ULONG:
    case DSP_UINT64:
    case DSP_INT64:
        *val = malloc(8);
        return (hg_proc_raw(proc, *val, 8));
    case DSP_STR:
    case DSP_JSON:
        ret = hg_proc_hg_string_t(proc, val);
        return (ret);
    default:
        return (HG_INVALID_PARAM);
    }
}

static inline hg_return_t hg_proc_dsp_dict_t(hg_proc_t proc, void *arg)
{
    hg_return_t ret = HG_SUCCESS;
    dsp_dict_t *in = (dsp_dict_t *)arg;
    int i;

    ret = hg_proc_hg_size_t(proc, &in->len);
    if(ret != HG_SUCCESS)
        return ret;

    if(hg_proc_get_op(proc) == HG_DECODE) {
        in->types = malloc(sizeof(*in->types) * in->len);
        in->keys = malloc(sizeof(*in->keys) * in->len);
        in->vals = malloc(sizeof(*in->vals) * in->len);
    }
    for(i = 0; i < in->len; i++) {
        ret = hg_proc_hg_int8_t(proc, &in->types[i]);
        if(ret != HG_SUCCESS) {
            return (ret);
        }
        ret = hg_proc_hg_string_t(proc, &in->keys[i]);
        if(ret != HG_SUCCESS) {
            return (ret);
        }
        if(hg_proc_get_op(proc) == HG_ENCODE) {
            ret = dict_encode_val(proc, in->types[i], in->vals[i]);
        } else if(hg_proc_get_op(proc) == HG_DECODE) {
            ret = dict_decode_val(proc, in->types[i], &in->vals[i]);
        } else {
            free(in->vals[i]);
        }
        if(ret != HG_SUCCESS)
            return (ret);
    }
    if(hg_proc_get_op(proc) == HG_FREE) {
        free(in->keys);
        free(in->vals);
    }

    return (ret);
}

MERCURY_GEN_PROC(bulk_gdim_t, ((odsc_hdr_with_gdim)(odsc))((hg_bulk_t)(handle)))
MERCURY_GEN_PROC(bulk_in_t,
                 ((odsc_hdr)(odsc))((hg_bulk_t)(handle))((uint8_t)(flags)))
MERCURY_GEN_PROC(bulk_out_t, ((int32_t)(ret))((hg_size_t)(len)))
MERCURY_GEN_PROC(put_meta_in_t, ((hg_string_t)(name))((int32_t)(length))(
                                    (int32_t)(version))((hg_bulk_t)(handle)))
MERCURY_GEN_PROC(query_meta_in_t,
                 ((hg_string_t)(name))((int32_t)(version))((uint8_t)(mode)))
MERCURY_GEN_PROC(query_meta_out_t, ((dsp_buf_t)(mdata))((int32_t)(version)))
MERCURY_GEN_PROC(peek_meta_in_t, ((hg_string_t)(name)))
MERCURY_GEN_PROC(peek_meta_out_t, ((int32_t)(res)))
MERCURY_GEN_PROC(odsc_gdim_t,
                 ((odsc_hdr_with_gdim)(odsc_gdim))((int32_t)(param)))
MERCURY_GEN_PROC(odsc_list_t, ((odsc_hdr)(odsc_list))((int32_t)(param)))
MERCURY_GEN_PROC(pexec_in_t, ((odsc_hdr)(odsc))((hg_bulk_t)(handle))(
                                 (int32_t)(length))((hg_string_t)(fn_name)))
MERCURY_GEN_PROC(pexec_out_t, ((hg_bulk_t)(handle))((int32_t)(length))(
                                  (uint64_t)(mtxp))((uint64_t)(condp)))
MERCURY_GEN_PROC(ss_information, ((odsc_hdr)(ss_buf))((hg_string_t)(chk_str)))
MERCURY_GEN_PROC(cond_in_t, ((uint64_t)(mtxp))((uint64_t)(condp)))
MERCURY_GEN_PROC(get_var_objs_in_t, ((hg_string_t)(var_name))((int32_t)(src)))
MERCURY_GEN_PROC(reg_in_t,
                 ((hg_string_t)(type))((hg_string_t)(name))(
                     (hg_string_t)(reg_data))((int32_t)(src))((uint64_t)(id)))
MERCURY_GEN_PROC(get_mod_in_t, ((hg_string_t)(name))((dsp_dict_t)(params)))
MERCURY_GEN_PROC(get_mod_out_t, ((odsc_list_t)(odscs))((int32_t)(ret)))

char *obj_desc_sprint(obj_descriptor *);
//
struct sspace *ssd_alloc(const struct bbox *, int, int,
                         enum sspace_hash_version);
struct sspace *ssd_alloc_v2(const struct bbox *bb_domain, int num_nodes,
                            int max_versions);
int ssd_init(struct sspace *, int);
void ssd_free(struct sspace *);
//

int ssd_copy(struct obj_data *, struct obj_data *);
//
long ssh_hash_elem_count(struct sspace *ss, const struct bbox *bb);
//
int ssd_filter(struct obj_data *, obj_descriptor *, double *);
int ssd_hash(struct sspace *, const struct bbox *, struct dht_entry *[]);

int dht_update_owner(struct dht_entry *de, obj_descriptor *odsc,
                     int clear_flag);
int dht_add_entry(struct dht_entry *, obj_descriptor *);
obj_descriptor *dht_find_entry(struct dht_entry *, obj_descriptor *);
int dht_find_entry_all(struct dht_entry *, obj_descriptor *,
                       obj_descriptor **[], int);
int dht_find_versions(struct dht_entry *, obj_descriptor *, int[]);
//
struct meta_data *meta_find_entry(ss_storage *ls, const char *name, int version,
                                  int wait);
struct meta_data *meta_find_next_entry(ss_storage *ls, const char *name,
                                       int curr, int wait);

ss_storage *ls_alloc(int max_versions);
void ls_free(ss_storage *);
void ls_add_meta(ss_storage *ls, struct meta_data *mdata);
void ls_add_obj(ss_storage *, struct obj_data *);
struct obj_data *ls_lookup(ss_storage *, char *);
void ls_remove(ss_storage *, struct obj_data *);
void ls_try_remove_free(ss_storage *, struct obj_data *);
struct obj_data *ls_find(ss_storage *, obj_descriptor *);
struct obj_data *ls_find_od(ss_storage *, obj_descriptor *);
int ls_find_ods(ss_storage *ls, obj_descriptor *odsc, obj_descriptor ***od_tab);
struct obj_data *ls_find_no_version(ss_storage *, obj_descriptor *);
int ls_get_var_names(ss_storage *, char ***);
int ls_find_all_no_version(ss_storage *ls, const char *var_name,
                           obj_descriptor ***odscs);
int ls_find_all(ss_storage *ls, obj_descriptor *odsc, struct obj_data ***ods);

struct obj_data *obj_data_alloc(obj_descriptor *);
struct obj_data *obj_data_alloc_no_data(obj_descriptor *, void *);
struct obj_data *obj_data_alloc_with_data(obj_descriptor *, void *);

void meta_data_free(struct meta_data *mdata);

void obj_data_free(struct obj_data *od);
uint64_t obj_data_size(obj_descriptor *);
void obj_data_resize(obj_descriptor *obj_desc, uint64_t *new_dims);

int obj_desc_equals(obj_descriptor *, obj_descriptor *);
int obj_desc_equals_no_owner(const obj_descriptor *, const obj_descriptor *);

int obj_desc_equals_intersect(obj_descriptor *odsc1, obj_descriptor *odsc2);

int obj_desc_by_name_intersect(const obj_descriptor *odsc1,
                               const obj_descriptor *odsc2);

// void copy_global_dimension(struct global_dimension *l, int ndim, const
// uint64_t *gdim);
int global_dimension_equal(const struct global_dimension *gdim1,
                           const struct global_dimension *gdim2);
void init_gdim_list(struct list_head *gdim_list);
void update_gdim_list(struct list_head *gdim_list, const char *var_name,
                      int ndim, uint64_t *gdim);
struct gdim_list_entry *lookup_gdim_list(struct list_head *gdim_list,
                                         const char *var_name);
void free_gdim_list(struct list_head *gdim_list);
void set_global_dimension(struct list_head *gdim_list, const char *var_name,
                          const struct global_dimension *default_gdim,
                          struct global_dimension *gdim);
void get_global_dimensions(struct global_dimension *l, int *ndim,
                           uint64_t *gdim);
void get_gdims(struct list_head *gdim_list, const char *var_name, int *ndim,
               uint64_t **gdim);

struct lock_data *get_lock(struct list_head *list, char *name);
struct lock_data *create_lock(struct list_head *list, char *name);

char **addr_str_buf_to_list(char *buf, int num_addrs);

#endif /* __SS_DATA_H_ */