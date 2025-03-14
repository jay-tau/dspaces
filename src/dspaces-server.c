/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers
 * University
 *
 * See COPYRIGHT in top-level directory.
 */
#include "dspaces-server.h"
#include "dspaces-conf.h"
#include "dspaces-logging.h"
#include "dspaces-modules.h"
#include "dspaces-ops.h"
#include "dspaces-remote.h"
#include "dspaces-storage.h"
#include "dspaces.h"
#include "dspacesp.h"
#include "gspace.h"
#include "ss_data.h"
#include "str_hash.h"
#include "toml.h"
#include <abt.h>
#include <errno.h>
#include <fcntl.h>
#include <lz4.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef OPS_USE_OPENMP
#include <omp.h>
#endif

#ifdef DSPACES_HAVE_PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL dsm
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#endif

#ifdef HAVE_DRC
#include <rdmacred.h>
#endif /* HAVE_DRC */

#define DSPACES_DEFAULT_NUM_HANDLERS 4

#define xstr(s) str(s)
#define str(s) #s

// TODO !
// static enum storage_type st = column_major;

typedef enum obj_update_type { DS_OBJ_NEW, DS_OBJ_OWNER } obj_update_t;

int cond_num = 0;

struct addr_list_entry {
    struct list_head entry;
    char *addr;
};

struct dspaces_provider {
    struct ds_conf conf;
    struct list_head dirs;
    margo_instance_id mid;
    hg_id_t put_id;
    hg_id_t put_local_id;
    hg_id_t put_meta_id;
    hg_id_t query_id;
    hg_id_t peek_meta_id;
    hg_id_t query_meta_id;
    hg_id_t get_id;
    hg_id_t get_local_id;
    hg_id_t obj_update_id;
    hg_id_t odsc_internal_id;
    hg_id_t ss_id;
    hg_id_t drain_id;
    hg_id_t kill_id;
    hg_id_t kill_client_id;
    hg_id_t sub_id;
    hg_id_t notify_id;
    hg_id_t do_ops_id;
    hg_id_t pexec_id;
    hg_id_t mpexec_id;
    hg_id_t cond_id;
    hg_id_t get_vars_id;
    hg_id_t get_var_objs_id;
    hg_id_t reg_id;
    hg_id_t get_mods_id;
    hg_id_t get_mod_id;
    struct list_head mods;
    struct ds_gspace *dsg;
    char **server_address;
    char **node_names;
    char *listen_addr_str;
    int rank;
    int comm_size;
    int f_debug;
    int f_drain;
    int f_kill;

#ifdef HAVE_DRC
    uint32_t drc_credential_id;
#endif

    MPI_Comm comm;

    ABT_mutex odsc_mutex;
    ABT_mutex ls_mutex;
    ABT_mutex dht_mutex;
    ABT_mutex sspace_mutex;
    ABT_mutex kill_mutex;

    ABT_xstream drain_xstream;
    ABT_pool drain_pool;
    ABT_thread drain_t;

    const char *pub_ip;
    const char *priv_ip;

    struct remote *remotes;
    int nremote;

    int num_handlers;
    int *handler_rmap;

    int local_reg_id;
#ifdef DSPACES_HAVE_PYTHON
    PyThreadState *main_state;
    PyThreadState **handler_state;
#endif // DSPACES_HAVE_PYTHON
};

static int dspaces_init_registry(dspaces_provider_t server);

DECLARE_MARGO_RPC_HANDLER(put_rpc)
DECLARE_MARGO_RPC_HANDLER(put_local_rpc)
DECLARE_MARGO_RPC_HANDLER(put_meta_rpc)
DECLARE_MARGO_RPC_HANDLER(get_rpc)
DECLARE_MARGO_RPC_HANDLER(query_rpc)
DECLARE_MARGO_RPC_HANDLER(peek_meta_rpc)
DECLARE_MARGO_RPC_HANDLER(query_meta_rpc)
DECLARE_MARGO_RPC_HANDLER(obj_update_rpc)
DECLARE_MARGO_RPC_HANDLER(odsc_internal_rpc)
DECLARE_MARGO_RPC_HANDLER(ss_rpc)
DECLARE_MARGO_RPC_HANDLER(kill_rpc)
DECLARE_MARGO_RPC_HANDLER(sub_rpc)
DECLARE_MARGO_RPC_HANDLER(do_ops_rpc)
#ifdef DSPACES_HAVE_PYTHON
DECLARE_MARGO_RPC_HANDLER(pexec_rpc)
DECLARE_MARGO_RPC_HANDLER(mpexec_rpc)
#endif // DSPACES_HAVE_PYTHON
DECLARE_MARGO_RPC_HANDLER(cond_rpc)
DECLARE_MARGO_RPC_HANDLER(get_vars_rpc);
DECLARE_MARGO_RPC_HANDLER(get_var_objs_rpc);
DECLARE_MARGO_RPC_HANDLER(reg_rpc);
DECLARE_MARGO_RPC_HANDLER(get_mods_rpc);
DECLARE_MARGO_RPC_HANDLER(get_mod_rpc);

static int init_sspace(dspaces_provider_t server, struct bbox *default_domain,
                       struct ds_gspace *dsg_l)
{
    int err = -ENOMEM;
    dsg_l->ssd =
        ssd_alloc(default_domain, dsg_l->size_sp, server->conf.max_versions,
                  server->conf.hash_version);
    if(!dsg_l->ssd)
        goto err_out;

    if(server->conf.hash_version == ssd_hash_version_auto) {
        DEBUG_OUT("server selected hash type %s for default space\n",
                  hash_strings[dsg_l->ssd->hash_version]);
    }

    err = ssd_init(dsg_l->ssd, dsg_l->rank);
    if(err < 0)
        goto err_out;

    dsg_l->default_gdim.ndim = server->conf.ndim;
    int i;
    for(i = 0; i < server->conf.ndim; i++) {
        dsg_l->default_gdim.sizes.c[i] = server->conf.dims.c[i];
    }

    INIT_LIST_HEAD(&dsg_l->sspace_list);
    return 0;
err_out:
    fprintf(stderr, "%s(): ERROR failed\n", __func__);
    return err;
}

static int write_conf(dspaces_provider_t server, MPI_Comm comm)
{
    hg_addr_t my_addr = HG_ADDR_NULL;
    char *my_addr_str = NULL;
    char my_node_str[HOST_NAME_MAX];
    hg_size_t my_addr_size = 0;
    int my_node_name_len = 0;
    int *str_sizes;
    hg_return_t hret = HG_SUCCESS;
    int buf_size = 0;
    int *sizes_psum;
    char *str_buf;
    FILE *fd;
    int i;
    int ret = 0;

    hret = margo_addr_self(server->mid, &my_addr);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_addr_self() returned %d\n",
                __func__, hret);
        ret = -1;
        goto error;
    }

    hret = margo_addr_to_string(server->mid, NULL, &my_addr_size, my_addr);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_addr_to_string() returned %d\n",
                __func__, hret);
        ret = -1;
        goto errorfree;
    }

    my_addr_str = malloc(my_addr_size);
    hret =
        margo_addr_to_string(server->mid, my_addr_str, &my_addr_size, my_addr);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_addr_to_string() returned %d\n",
                __func__, hret);
        ret = -1;
        goto errorfree;
    }

    MPI_Comm_size(comm, &server->comm_size);
    str_sizes = malloc(server->comm_size * sizeof(*str_sizes));
    sizes_psum = malloc(server->comm_size * sizeof(*sizes_psum));
    // TODO: MPI_Reduce instead
    MPI_Allgather(&my_addr_size, 1, MPI_INT, str_sizes, 1, MPI_INT, comm);
    sizes_psum[0] = 0;
    for(i = 0; i < server->comm_size; i++) {
        buf_size += str_sizes[i];
        if(i) {
            sizes_psum[i] = sizes_psum[i - 1] + str_sizes[i - 1];
        }
    }
    if(buf_size > 0) {
        str_buf = malloc(buf_size);
        MPI_Allgatherv(my_addr_str, my_addr_size, MPI_CHAR, str_buf, str_sizes,
                       sizes_psum, MPI_CHAR, comm);
    }
    server->server_address =
        malloc(server->comm_size * sizeof(*server->server_address));
    for(i = 0; i < server->comm_size; i++) {
        server->server_address[i] = &str_buf[sizes_psum[i]];
    }

    gethostname(my_node_str, HOST_NAME_MAX);
    my_node_str[HOST_NAME_MAX - 1] = '\0';
    my_node_name_len = strlen(my_node_str) + 1;
    MPI_Allgather(&my_node_name_len, 1, MPI_INT, str_sizes, 1, MPI_INT, comm);
    sizes_psum[0] = 0;
    buf_size = 0;
    for(i = 0; i < server->comm_size; i++) {
        buf_size += str_sizes[i];
        if(i) {
            sizes_psum[i] = sizes_psum[i - 1] + str_sizes[i - 1];
        }
    }
    str_buf = malloc(buf_size);
    MPI_Allgatherv(my_node_str, my_node_name_len, MPI_CHAR, str_buf, str_sizes,
                   sizes_psum, MPI_CHAR, comm);
    server->node_names =
        malloc(server->comm_size * sizeof(*server->node_names));
    for(i = 0; i < server->comm_size; i++) {
        server->node_names[i] = &str_buf[sizes_psum[i]];
    }

    MPI_Comm_rank(comm, &server->rank);
    if(server->rank == 0) {
        fd = fopen("conf.ds", "w");
        if(!fd) {
            fprintf(stderr,
                    "ERROR: %s: unable to open 'conf.ds' for writing.\n",
                    __func__);
            ret = -1;
            goto errorfree;
        }
        fprintf(fd, "%d\n", server->comm_size);
        for(i = 0; i < server->comm_size; i++) {
            fprintf(fd, "%s %s\n", server->node_names[i],
                    server->server_address[i]);
        }
        fprintf(fd, "%s\n", server->listen_addr_str);
#ifdef HAVE_DRC
        fprintf(fd, "%" PRIu32 "\n", server->drc_credential_id);
#endif
        fclose(fd);
    }

    free(my_addr_str);
    free(str_sizes);
    free(sizes_psum);
    margo_addr_free(server->mid, my_addr);

    return (ret);

errorfree:
    margo_addr_free(server->mid, my_addr);
error:
    margo_finalize(server->mid);
    return (ret);
}

static int dsg_alloc(dspaces_provider_t server, const char *conf_name,
                     MPI_Comm comm)
{
    struct ds_gspace *dsg_l;
    char *ext;
    int err = -ENOMEM;

    /* Default values */
    server->conf.max_versions = 255;
    server->conf.hash_version = ssd_hash_version_auto;
    server->conf.num_apps = -1;

    INIT_LIST_HEAD(&server->dirs);
    INIT_LIST_HEAD(&server->mods);

    // TODO: distribute configuration from root.

    ext = strrchr(conf_name, '.');
    if(!ext || strcmp(ext, ".toml") != 0) {
        err = parse_conf(conf_name, &server->conf);
    } else {
        server->conf.dirs = &server->dirs;
        server->conf.remotes = &server->remotes;
        server->conf.mods = &server->mods;
        err = parse_conf_toml(conf_name, &server->conf);
        server->nremote = server->conf.nremote;
    }
    if(err < 0) {
        goto err_out;
    }

    // Check number of dimension
    if(server->conf.ndim > BBOX_MAX_NDIM) {
        fprintf(
            stderr,
            "%s(): ERROR maximum number of array dimension is %d but ndim is %d"
            " in file '%s'\n",
            __func__, BBOX_MAX_NDIM, server->conf.ndim, conf_name);
        err = -EINVAL;
        goto err_out;
    } else if(server->conf.ndim == 0) {
        DEBUG_OUT(
            "no global coordinates provided. Setting trivial placeholder.\n");
        server->conf.ndim = 1;
        server->conf.dims.c[0] = 1;
    }

    // Check hash version
    if((server->conf.hash_version < ssd_hash_version_auto) ||
       (server->conf.hash_version >= _ssd_hash_version_count)) {
        fprintf(stderr, "%s(): ERROR unknown hash version %d in file '%s'\n",
                __func__, server->conf.hash_version, conf_name);
        err = -EINVAL;
        goto err_out;
    }

    struct bbox domain;
    memset(&domain, 0, sizeof(struct bbox));
    domain.num_dims = server->conf.ndim;
    int i;
    for(i = 0; i < domain.num_dims; i++) {
        domain.lb.c[i] = 0;
        domain.ub.c[i] = server->conf.dims.c[i] - 1;
    }

    dsg_l = malloc(sizeof(*dsg_l));
    if(!dsg_l)
        goto err_out;

    MPI_Comm_size(comm, &(dsg_l->size_sp));

    MPI_Comm_rank(comm, &dsg_l->rank);

    if(dsg_l->rank == 0) {
        print_conf(&server->conf);
    }

    err = init_sspace(server, &domain, dsg_l);
    if(err < 0) {
        goto err_free;
    }
    dsg_l->ls = ls_alloc(server->conf.max_versions);
    if(!dsg_l->ls) {
        fprintf(stderr, "%s(): ERROR ls_alloc() failed\n", __func__);
        goto err_free;
    }

    // proxy storage
    dsg_l->ps = ls_alloc(server->conf.max_versions);
    if(!dsg_l->ps) {
        fprintf(stderr, "%s(): ERROR ls_alloc() failed\n", __func__);
        goto err_free;
    }

    dsg_l->num_apps = server->conf.num_apps;

    INIT_LIST_HEAD(&dsg_l->obj_desc_drain_list);

    server->dsg = dsg_l;

    write_conf(server, comm);

    return 0;
err_free:
    free(dsg_l);
err_out:
    fprintf(stderr, "'%s()': failed with %d.\n", __func__, err);
    return err;
}

static int free_sspace(struct ds_gspace *dsg_l)
{
    ssd_free(dsg_l->ssd);
    struct sspace_list_entry *ssd_entry, *temp;
    list_for_each_entry_safe(ssd_entry, temp, &dsg_l->sspace_list,
                             struct sspace_list_entry, entry)
    {
        ssd_free(ssd_entry->ssd);
        list_del(&ssd_entry->entry);
        free(ssd_entry);
    }

    return 0;
}

static struct sspace *lookup_sspace(dspaces_provider_t server,
                                    const char *var_name,
                                    const struct global_dimension *gd)
{
    struct global_dimension gdim;
    struct ds_gspace *dsg_l = server->dsg;
    int i;

    memcpy(&gdim, gd, sizeof(struct global_dimension));

    if(server->f_debug) {
        DEBUG_OUT("global dimensions for %s:\n", var_name);
        for(i = 0; i < gdim.ndim; i++) {
            DEBUG_OUT(" dim[%i] = %" PRIu64 "\n", i, gdim.sizes.c[i]);
        }
    }

    // Return the default shared space created based on
    // global data domain specified in dataspaces.conf
    if(global_dimension_equal(&gdim, &dsg_l->default_gdim)) {
        DEBUG_OUT("uses default gdim\n");
        return dsg_l->ssd;
    }

    // Otherwise, search for shared space based on the
    // global data domain specified by application in put()/get().
    struct sspace_list_entry *ssd_entry = NULL;
    list_for_each_entry(ssd_entry, &dsg_l->sspace_list,
                        struct sspace_list_entry, entry)
    {
        // compare global dimension
        if(gdim.ndim != ssd_entry->gdim.ndim)
            continue;

        if(global_dimension_equal(&gdim, &ssd_entry->gdim))
            return ssd_entry->ssd;
    }

    DEBUG_OUT("didn't find an existing shared space. Make a new one.\n");

    // If not found, add new shared space
    int err;
    struct bbox domain;
    memset(&domain, 0, sizeof(struct bbox));
    domain.num_dims = gdim.ndim;
    DEBUG_OUT("global dimmensions being allocated:\n");
    for(i = 0; i < gdim.ndim; i++) {
        domain.lb.c[i] = 0;
        domain.ub.c[i] = gdim.sizes.c[i] - 1;
        DEBUG_OUT("dim %i: lb = %" PRIu64 ", ub = %" PRIu64 "\n", i,
                  domain.lb.c[i], domain.ub.c[i]);
    }

    ssd_entry = malloc(sizeof(struct sspace_list_entry));
    memcpy(&ssd_entry->gdim, &gdim, sizeof(struct global_dimension));

    DEBUG_OUT("allocate the ssd.\n");
    ssd_entry->ssd =
        ssd_alloc(&domain, dsg_l->size_sp, server->conf.max_versions,
                  server->conf.hash_version);
    if(!ssd_entry->ssd) {
        fprintf(stderr, "%s(): ssd_alloc failed for '%s'\n", __func__,
                var_name);
        return dsg_l->ssd;
    }

    if(server->conf.hash_version == ssd_hash_version_auto) {
        DEBUG_OUT("server selected hash version %i for var %s\n",
                  ssd_entry->ssd->hash_version, var_name);
    }

    DEBUG_OUT("doing ssd init\n");
    err = ssd_init(ssd_entry->ssd, dsg_l->rank);
    if(err < 0) {
        fprintf(stderr, "%s(): ssd_init failed\n", __func__);
        return dsg_l->ssd;
    }

    list_add(&ssd_entry->entry, &dsg_l->sspace_list);
    return ssd_entry->ssd;
}

static void obj_update_local_dht(dspaces_provider_t server,
                                 obj_descriptor *odsc, struct sspace *ssd,
                                 obj_update_t type)
{
    DEBUG_OUT("Add in local_dht %d\n", server->dsg->rank);
    ABT_mutex_lock(server->dht_mutex);
    switch(type) {
    case DS_OBJ_NEW:
        dht_add_entry(ssd->ent_self, odsc);
        break;
    case DS_OBJ_OWNER:
        dht_update_owner(ssd->ent_self, odsc, 1);
        break;
    default:
        fprintf(stderr, "ERROR: (%s): unknown object update type.\n", __func__);
    }
    ABT_mutex_unlock(server->dht_mutex);
}

static int obj_update_dht(dspaces_provider_t server, struct obj_data *od,
                          obj_update_t type)
{
    obj_descriptor *odsc = &od->obj_desc;
    DEBUG_OUT("getting sspace lock.\n");
    ABT_mutex_lock(server->sspace_mutex);
    DEBUG_OUT("got sspace lock.\n");
    struct sspace *ssd = lookup_sspace(server, odsc->name, &od->gdim);
    DEBUG_OUT("realeasing sspace lock.\n");
    ABT_mutex_unlock(server->sspace_mutex);
    struct dht_entry *dht_tab[ssd->dht->num_entries];

    int num_de, i;

    /* Compute object distribution to nodes in the space. */
    num_de = ssd_hash(ssd, &odsc->bb, dht_tab);
    if(num_de == 0) {
        DEBUG_OUT("Could not distribute the object in a spatial index. Storing "
                  "locally.\n");
        obj_update_local_dht(server, odsc, ssd, type);
    }

    for(i = 0; i < num_de; i++) {
        if(dht_tab[i]->rank == server->dsg->rank) {
            obj_update_local_dht(server, odsc, ssd, type);
            continue;
        }

        // now send rpc to the server for dht_update
        hg_return_t hret;
        odsc_gdim_t in;
        margo_request req;
        DEBUG_OUT("Server %d sending object %s to dht server %d \n",
                  server->dsg->rank, obj_desc_sprint(odsc), dht_tab[i]->rank);

        in.odsc_gdim.size = sizeof(*odsc);
        in.odsc_gdim.gdim_size = sizeof(struct global_dimension);
        in.odsc_gdim.raw_odsc = (char *)(odsc);
        in.odsc_gdim.raw_gdim = (char *)(&od->gdim);
        in.param = type;

        hg_addr_t svr_addr;
        margo_addr_lookup(server->mid, server->server_address[dht_tab[i]->rank],
                          &svr_addr);

        hg_handle_t h;
        margo_create(server->mid, svr_addr, server->obj_update_id, &h);
        margo_iforward(h, &in, &req);
        DEBUG_OUT("sent obj server %d to update dht %s in \n", dht_tab[i]->rank,
                  obj_desc_sprint(odsc));

        margo_addr_free(server->mid, svr_addr);
        hret = margo_destroy(h);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): could not destroy handle!\n",
                    __func__);
            return (dspaces_ERR_MERCURY);
        }
    }

    return dspaces_SUCCESS;
}

static int get_client_data(obj_descriptor odsc, dspaces_provider_t server)
{
    bulk_in_t in;
    bulk_out_t out;
    struct obj_data *od;
    int ret;

    od = obj_data_alloc(&odsc);
    in.odsc.size = sizeof(obj_descriptor);
    in.odsc.raw_odsc = (char *)(&odsc);

    hg_addr_t owner_addr;
    hg_size_t owner_addr_size = 128;

    margo_addr_self(server->mid, &owner_addr);
    margo_addr_to_string(server->mid, od->obj_desc.owner, &owner_addr_size,
                         owner_addr);
    margo_addr_free(server->mid, owner_addr);

    hg_size_t rdma_size = (odsc.size) * bbox_volume(&odsc.bb);

    margo_bulk_create(server->mid, 1, (void **)(&(od->data)), &rdma_size,
                      HG_BULK_WRITE_ONLY, &in.handle);
    hg_addr_t client_addr;
    margo_addr_lookup(server->mid, odsc.owner, &client_addr);

    hg_handle_t handle;
    margo_create(server->mid, client_addr, server->drain_id, &handle);
    margo_forward(handle, &in);
    margo_get_output(handle, &out);
    if(out.ret == dspaces_SUCCESS) {
        ABT_mutex_lock(server->ls_mutex);
        ls_add_obj(server->dsg->ls, od);
        ABT_mutex_unlock(server->ls_mutex);
    }
    ret = out.ret;
    // now update the dht with new owner information
    DEBUG_OUT("Inside get_client_data\n");
    margo_addr_free(server->mid, client_addr);
    margo_bulk_free(in.handle);
    margo_free_output(handle, &out);
    margo_destroy(handle);
    obj_update_dht(server, od, DS_OBJ_OWNER);
    return ret;
}

// thread to move data between layers
static void drain_thread(void *arg)
{
    dspaces_provider_t server = arg;

    while(server->f_kill != 0) {
        int counter = 0;
        DEBUG_OUT("Thread WOKEUP\n");
        do {
            counter = 0;
            obj_descriptor odsc;
            struct obj_desc_list *odscl;
            // requires better way to get the obj_descriptor
            ABT_mutex_lock(server->odsc_mutex);
            DEBUG_OUT("Inside odsc mutex\n");
            list_for_each_entry(odscl, &(server->dsg->obj_desc_drain_list),
                                struct obj_desc_list, odsc_entry)
            {
                memcpy(&odsc, &(odscl->odsc), sizeof(obj_descriptor));
                DEBUG_OUT("Found %s in odsc_list\n", obj_desc_sprint(&odsc));
                counter = 1;
                break;
            }
            if(counter == 1) {
                list_del(&odscl->odsc_entry);
                ABT_mutex_unlock(server->odsc_mutex);
                int ret = get_client_data(odsc, server);
                DEBUG_OUT("Finished draining %s\n", obj_desc_sprint(&odsc));
                if(ret != dspaces_SUCCESS) {
                    ABT_mutex_lock(server->odsc_mutex);
                    DEBUG_OUT("Drain failed, returning object to queue...\n");
                    list_add_tail(&odscl->odsc_entry,
                                  &server->dsg->obj_desc_drain_list);
                    ABT_mutex_unlock(server->odsc_mutex);
                }
                sleep(1);
            } else {
                ABT_mutex_unlock(server->odsc_mutex);
            }

        } while(counter == 1);

        sleep(10);

        ABT_thread_yield();
    }
}

#ifdef DSPACES_HAVE_PYTHON
static void *bootstrap_python(dspaces_provider_t server)
{
    char *pypath = getenv("PYTHONPATH");
    char *new_pypath;
    int pypath_len;
    int i;

    pypath_len = strlen(xstr(DSPACES_MOD_DIR)) + 1;
    if(pypath) {
        pypath_len += strlen(pypath) + 1;
    }

    new_pypath = malloc(pypath_len);
    if(pypath) {
        sprintf(new_pypath, "%s:%s", xstr(DSPACES_MOD_DIR), pypath);
    } else {
        strcpy(new_pypath, xstr(DSPACES_MOD_DIR));
    }
    setenv("PYTHONPATH", new_pypath, 1);
    DEBUG_OUT("New PYTHONPATH is %s\n", new_pypath);

    Py_InitializeEx(0);
    import_array();

    server->handler_state =
        malloc(sizeof(*server->handler_state) * server->num_handlers);

    return (NULL);
}

#endif // DSPACES_HAVE_PYTHON

static void init_rmap(struct dspaces_provider *server)
{
    int i;

    server->handler_rmap =
        malloc(sizeof(*server->handler_rmap) * server->num_handlers);
    for(i = 0; i < server->num_handlers; i++) {
        server->handler_rmap[i] = -1;
    }
}

static int get_handler_id(struct dspaces_provider *server)
{
    int es_rank;
    int i;

    ABT_self_get_xstream_rank(&es_rank);

    for(i = 0; i < server->num_handlers; i++) {
        if(server->handler_rmap[i] == -1) {
            server->handler_rmap[i] = es_rank;
        }

        if(server->handler_rmap[i] == es_rank) {
            return (i);
        }
    }

    fprintf(stderr,
            "ERROR: more unique execution streams have called %s than have "
            "been allocated for RPC handlers.\n",
            __func__);

    return (-1);
}

int dspaces_server_init(const char *listen_addr_str, MPI_Comm comm,
                        const char *conf_file, dspaces_provider_t *sv)
{
    const char *envdebug = getenv("DSPACES_DEBUG");
    const char *envnthreads = getenv("DSPACES_NUM_HANDLERS");
    const char *envdrain = getenv("DSPACES_DRAIN");
    const char *mod_dir_str = xstr(DSPACES_MOD_DIR);
    dspaces_provider_t server;
    hg_class_t *hg;
    static int is_initialized = 0;
    hg_bool_t flag;
    hg_id_t id;
    struct hg_init_info hii = {0};
    char margo_conf[1024];
    struct margo_init_info mii = {0};
    int i, ret;

    if(is_initialized) {
        fprintf(stderr,
                "DATASPACES: WARNING: %s: multiple instantiations of the "
                "dataspaces server is not supported.\n",
                __func__);
        return (dspaces_ERR_ALLOCATION);
    }

    server = (dspaces_provider_t)calloc(1, sizeof(*server));
    if(server == NULL)
        return dspaces_ERR_ALLOCATION;

    if(envdebug) {
        server->f_debug = 1;
    }

    server->num_handlers = DSPACES_DEFAULT_NUM_HANDLERS;
    if(envnthreads) {
        server->num_handlers = atoi(envnthreads);
    }
    init_rmap(server);

    if(envdrain) {
        DEBUG_OUT("enabling data draining.\n");
        server->f_drain = 1;
    }

    MPI_Comm_dup(comm, &server->comm);
    MPI_Comm_rank(comm, &server->rank);

    dspaces_init_logging(server->rank);

    margo_set_environment(NULL);
    sprintf(margo_conf,
            "{ \"use_progress_thread\" : true, \"rpc_thread_count\" : %d }",
            server->num_handlers);
    hii.request_post_init = 1024;
    hii.auto_sm = 0;
    mii.hg_init_info = &hii;
    mii.json_config = margo_conf;
    ABT_init(0, NULL);

#ifdef HAVE_DRC

    server->drc_credential_id = 0;
    if(server->rank == 0) {
        ret =
            drc_acquire(&server->drc_credential_id, DRC_FLAGS_FLEX_CREDENTIAL);
        if(ret != DRC_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): drc_acquire failure %d\n", __func__,
                    ret);
            return ret;
        }
    }
    MPI_Bcast(&server->drc_credential_id, 1, MPI_UINT32_T, 0, comm);

    /* access credential on all ranks and convert to string for use by mercury
     */

    drc_info_handle_t drc_credential_info;
    uint32_t drc_cookie;
    char drc_key_str[256] = {0};

    ret = drc_access(server->drc_credential_id, 0, &drc_credential_info);
    if(ret != DRC_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): drc_access failure %d\n", __func__, ret);
        return ret;
    }

    drc_cookie = drc_get_first_cookie(drc_credential_info);
    sprintf(drc_key_str, "%u", drc_cookie);

    memset(&hii, 0, sizeof(hii));
    hii.na_init_info.auth_key = drc_key_str;

    /* rank 0 grants access to the credential, allowing other jobs to use it */
    if(server->rank == 0) {
        ret = drc_grant(server->drc_credential_id, drc_get_wlm_id(),
                        DRC_FLAGS_TARGET_WLM);
        if(ret != DRC_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): drc_grants failure %d\n", __func__,
                    ret);
            return ret;
        }
    }

    server->mid = margo_init_ext(listen_addr_str, MARGO_SERVER_MODE, &mii);

#else

    server->mid = margo_init_ext(listen_addr_str, MARGO_SERVER_MODE, &mii);
    if(server->f_debug) {
        if(!server->rank) {
            char *margo_json = margo_get_config(server->mid);
            fprintf(stderr, "%s", margo_json);
            free(margo_json);
        }
        margo_set_log_level(server->mid, MARGO_LOG_WARNING);
    }
    MPI_Barrier(comm);

#endif /* HAVE_DRC */
    DEBUG_OUT("did margo init\n");
    if(!server->mid) {
        fprintf(stderr, "ERROR: %s: margo_init() failed.\n", __func__);
        return (dspaces_ERR_MERCURY);
    }
    server->listen_addr_str = strdup(listen_addr_str);

    ABT_mutex_create(&server->odsc_mutex);
    ABT_mutex_create(&server->ls_mutex);
    ABT_mutex_create(&server->dht_mutex);
    ABT_mutex_create(&server->sspace_mutex);
    ABT_mutex_create(&server->kill_mutex);

    hg = margo_get_class(server->mid);

    margo_registered_name(server->mid, "put_rpc", &id, &flag);

    if(flag == HG_TRUE) { /* RPCs already registered */
        DEBUG_OUT("RPC names already registered. Setting handlers...\n");
        margo_registered_name(server->mid, "put_rpc", &server->put_id, &flag);
        DS_HG_REGISTER(hg, server->put_id, bulk_gdim_t, bulk_out_t, put_rpc);
        margo_registered_name(server->mid, "put_local_rpc",
                              &server->put_local_id, &flag);
        DS_HG_REGISTER(hg, server->put_local_id, odsc_gdim_t, bulk_out_t,
                       put_local_rpc);
        margo_registered_name(server->mid, "put_meta_rpc", &server->put_meta_id,
                              &flag);
        DS_HG_REGISTER(hg, server->put_meta_id, put_meta_in_t, bulk_out_t,
                       put_meta_rpc);
        margo_registered_name(server->mid, "get_rpc", &server->get_id, &flag);
        DS_HG_REGISTER(hg, server->get_id, bulk_in_t, bulk_out_t, get_rpc);
        margo_registered_name(server->mid, "get_local_rpc",
                              &server->get_local_id, &flag);
        margo_registered_name(server->mid, "query_rpc", &server->query_id,
                              &flag);
        DS_HG_REGISTER(hg, server->query_id, odsc_gdim_t, odsc_list_t,
                       query_rpc);
        margo_registered_name(server->mid, "peek_meta_rpc",
                              &server->peek_meta_id, &flag);
        DS_HG_REGISTER(hg, server->peek_meta_id, peek_meta_in_t,
                       peek_meta_out_t, peek_meta_rpc);
        margo_registered_name(server->mid, "query_meta_rpc",
                              &server->query_meta_id, &flag);
        DS_HG_REGISTER(hg, server->query_meta_id, query_meta_in_t,
                       query_meta_out_t, query_meta_rpc);
        margo_registered_name(server->mid, "obj_update_rpc",
                              &server->obj_update_id, &flag);
        DS_HG_REGISTER(hg, server->obj_update_id, odsc_gdim_t, void,
                       obj_update_rpc);
        margo_registered_name(server->mid, "odsc_internal_rpc",
                              &server->odsc_internal_id, &flag);
        DS_HG_REGISTER(hg, server->odsc_internal_id, odsc_gdim_t, odsc_list_t,
                       odsc_internal_rpc);
        margo_registered_name(server->mid, "ss_rpc", &server->ss_id, &flag);
        DS_HG_REGISTER(hg, server->ss_id, void, ss_information, ss_rpc);
        margo_registered_name(server->mid, "drain_rpc", &server->drain_id,
                              &flag);
        margo_registered_name(server->mid, "kill_rpc", &server->kill_id, &flag);
        DS_HG_REGISTER(hg, server->kill_id, int32_t, void, kill_rpc);
        margo_registered_name(server->mid, "kill_client_rpc",
                              &server->kill_client_id, &flag);
        margo_registered_name(server->mid, "sub_rpc", &server->sub_id, &flag);
        DS_HG_REGISTER(hg, server->sub_id, odsc_gdim_t, void, sub_rpc);
        margo_registered_name(server->mid, "notify_rpc", &server->notify_id,
                              &flag);
        margo_registered_name(server->mid, "do_ops_rpc", &server->do_ops_id,
                              &flag);
        DS_HG_REGISTER(hg, server->do_ops_id, do_ops_in_t, bulk_out_t,
                       do_ops_rpc);
#ifdef DSPACES_HAVE_PYTHON
        margo_registered_name(server->mid, "pexec_rpc", &server->pexec_id,
                              &flag);
        DS_HG_REGISTER(hg, server->pexec_id, pexec_in_t, pexec_out_t,
                       pexec_rpc);
        margo_registered_name(server->mid, "mpexec_rpc", &server->mpexec_id,
                              &flag);
        DS_HG_REGISTER(hg, server->mpexec_id, pexec_in_t, pexec_out_t,
                       pexec_rpc);
#endif // DSPACES_HAVE_PYTHON
        margo_registered_name(server->mid, "cond_rpc", &server->cond_id, &flag);
        DS_HG_REGISTER(hg, server->cond_id, cond_in_t, void, cond_rpc);
        margo_registered_name(server->mid, "get_vars_rpc", &server->get_vars_id,
                              &flag);
        DS_HG_REGISTER(hg, server->get_vars_id, int32_t, name_list_t,
                       get_vars_rpc);
        margo_registered_name(server->mid, "get_var_objs_rpc",
                              &server->get_var_objs_id, &flag);
        DS_HG_REGISTER(hg, server->get_var_objs_id, get_var_objs_in_t, odsc_hdr,
                       get_var_objs_rpc);
        margo_registered_name(server->mid, "reg_rpc", &server->reg_id, &flag);
        DS_HG_REGISTER(hg, server->reg_id, reg_in_t, uint64_t, reg_rpc);
        margo_registered_name(server->mid, "get_mods_rpc", &server->get_mods_id,
                              &flag);
        DS_HG_REGISTER(hg, server->get_mods_id, void, name_list_t,
                       get_mods_rpc);
        margo_registered_name(server->mid, "get_mod_rpc", &server->get_mod_id,
                              &flag);
        DS_HG_REGISTER(hg, server->get_mod_id, get_mod_in_t, get_mod_out_t,
                       get_mod_rpc);
    } else {
        server->put_id = MARGO_REGISTER(server->mid, "put_rpc", bulk_gdim_t,
                                        bulk_out_t, put_rpc);
        margo_register_data(server->mid, server->put_id, (void *)server, NULL);
        server->put_local_id =
            MARGO_REGISTER(server->mid, "put_local_rpc", odsc_gdim_t,
                           bulk_out_t, put_local_rpc);
        margo_register_data(server->mid, server->put_local_id, (void *)server,
                            NULL);
        server->put_meta_id =
            MARGO_REGISTER(server->mid, "put_meta_rpc", put_meta_in_t,
                           bulk_out_t, put_meta_rpc);
        margo_register_data(server->mid, server->put_meta_id, (void *)server,
                            NULL);
        server->get_id = MARGO_REGISTER(server->mid, "get_rpc", bulk_in_t,
                                        bulk_out_t, get_rpc);
        server->get_local_id = MARGO_REGISTER(server->mid, "get_local_rpc",
                                              bulk_in_t, bulk_out_t, NULL);
        margo_register_data(server->mid, server->get_id, (void *)server, NULL);
        server->query_id = MARGO_REGISTER(server->mid, "query_rpc", odsc_gdim_t,
                                          odsc_list_t, query_rpc);
        margo_register_data(server->mid, server->query_id, (void *)server,
                            NULL);
        server->peek_meta_id =
            MARGO_REGISTER(server->mid, "peek_meta_rpc", peek_meta_in_t,
                           peek_meta_out_t, peek_meta_rpc);
        margo_register_data(server->mid, server->peek_meta_id, (void *)server,
                            NULL);
        server->query_meta_id =
            MARGO_REGISTER(server->mid, "query_meta_rpc", query_meta_in_t,
                           query_meta_out_t, query_meta_rpc);
        margo_register_data(server->mid, server->query_meta_id, (void *)server,
                            NULL);
        server->obj_update_id = MARGO_REGISTER(
            server->mid, "obj_update_rpc", odsc_gdim_t, void, obj_update_rpc);
        margo_register_data(server->mid, server->obj_update_id, (void *)server,
                            NULL);
        margo_registered_disable_response(server->mid, server->obj_update_id,
                                          HG_TRUE);
        server->odsc_internal_id =
            MARGO_REGISTER(server->mid, "odsc_internal_rpc", odsc_gdim_t,
                           odsc_list_t, odsc_internal_rpc);
        margo_register_data(server->mid, server->odsc_internal_id,
                            (void *)server, NULL);
        server->ss_id =
            MARGO_REGISTER(server->mid, "ss_rpc", void, ss_information, ss_rpc);
        margo_register_data(server->mid, server->ss_id, (void *)server, NULL);
        server->drain_id = MARGO_REGISTER(server->mid, "drain_rpc", bulk_in_t,
                                          bulk_out_t, NULL);
        server->kill_id =
            MARGO_REGISTER(server->mid, "kill_rpc", int32_t, void, kill_rpc);
        margo_registered_disable_response(server->mid, server->kill_id,
                                          HG_TRUE);
        margo_register_data(server->mid, server->kill_id, (void *)server, NULL);
        server->kill_client_id =
            MARGO_REGISTER(server->mid, "kill_client_rpc", int32_t, void, NULL);
        margo_registered_disable_response(server->mid, server->kill_client_id,
                                          HG_TRUE);
        server->sub_id =
            MARGO_REGISTER(server->mid, "sub_rpc", odsc_gdim_t, void, sub_rpc);
        margo_register_data(server->mid, server->sub_id, (void *)server, NULL);
        margo_registered_disable_response(server->mid, server->sub_id, HG_TRUE);
        server->notify_id =
            MARGO_REGISTER(server->mid, "notify_rpc", odsc_list_t, void, NULL);
        margo_registered_disable_response(server->mid, server->notify_id,
                                          HG_TRUE);
        server->do_ops_id = MARGO_REGISTER(server->mid, "do_ops_rpc",
                                           do_ops_in_t, bulk_out_t, do_ops_rpc);
        margo_register_data(server->mid, server->do_ops_id, (void *)server,
                            NULL);
#ifdef DSPACES_HAVE_PYTHON
        server->pexec_id = MARGO_REGISTER(server->mid, "pexec_rpc", pexec_in_t,
                                          pexec_out_t, pexec_rpc);
        margo_register_data(server->mid, server->pexec_id, (void *)server,
                            NULL);
        server->mpexec_id = MARGO_REGISTER(server->mid, "mpexec_rpc",
                                           pexec_in_t, pexec_out_t, mpexec_rpc);
        margo_register_data(server->mid, server->mpexec_id, (void *)server,
                            NULL);
#endif // DSPACES_HAVE_PYTHON
        server->cond_id =
            MARGO_REGISTER(server->mid, "cond_rpc", cond_in_t, void, cond_rpc);
        margo_register_data(server->mid, server->cond_id, (void *)server, NULL);
        margo_registered_disable_response(server->mid, server->cond_id,
                                          HG_TRUE);
        server->get_vars_id = MARGO_REGISTER(
            server->mid, "get_vars_rpc", int32_t, name_list_t, get_vars_rpc);
        margo_register_data(server->mid, server->get_vars_id, (void *)server,
                            NULL);
        server->get_var_objs_id =
            MARGO_REGISTER(server->mid, "get_var_objs_rpc", get_var_objs_in_t,
                           odsc_hdr, get_var_objs_rpc);
        margo_register_data(server->mid, server->get_var_objs_id,
                            (void *)server, NULL);
        server->reg_id =
            MARGO_REGISTER(server->mid, "reg_rpc", reg_in_t, uint64_t, reg_rpc);
        margo_register_data(server->mid, server->reg_id, (void *)server, NULL);

        server->get_mods_id = MARGO_REGISTER(server->mid, "get_mods_rpc", void,
                                             name_list_t, get_mods_rpc);
        margo_register_data(server->mid, server->get_mods_id, (void *)server,
                            NULL);
        server->get_mod_id =
            MARGO_REGISTER(server->mid, "get_mod_rpc", get_mod_in_t,
                           get_mod_out_t, get_mod_rpc);
        margo_register_data(server->mid, server->get_mod_id, (void *)server,
                            NULL);
    }
    int err = dsg_alloc(server, conf_file, comm);
    if(err) {
        fprintf(stderr,
                "DATASPACES: ERROR: %s: could not allocate internal "
                "structures. (%d)\n",
                __func__, err);
        return (dspaces_ERR_ALLOCATION);
    }
    for(i = 0; i < server->nremote; i++) {
        DEBUG_OUT("initializing client connection to %s\n",
                  server->remotes[i].name);
        dspaces_init_wan(listen_addr_str, server->remotes[i].addr_str, 0,
                         &server->remotes[i].conn);
    }

    server->f_kill = server->dsg->num_apps;
    if(server->f_kill > 0) {
        DEBUG_OUT("Server will wait for %i kill tokens before halting.\n",
                  server->f_kill);
    } else {
        DEBUG_OUT("Server will run indefinitely.\n");
    }

#ifdef DSPACES_HAVE_PYTHON
    bootstrap_python(server);
#endif // DSPACES_HAVE_PYTHON

    DEBUG_OUT("module directory is %s\n", mod_dir_str);
    dspaces_init_mods(&server->mods);
    dspaces_init_registry(server);
#ifdef DSPACES_HAVE_PYTHON
    server->main_state = PyEval_SaveThread();
#endif // DSPACES_HAVE_PYTHON

    if(server->f_drain) {
        // thread to drain the data
        ABT_xstream_create(ABT_SCHED_NULL, &server->drain_xstream);
        ABT_xstream_get_main_pools(server->drain_xstream, 1,
                                   &server->drain_pool);
        ABT_thread_create(server->drain_pool, drain_thread, server,
                          ABT_THREAD_ATTR_NULL, &server->drain_t);
    }

    server->pub_ip = getenv("DSPACES_PUBLIC_IP");
    server->priv_ip = getenv("DSPACES_PRIVATE_IP");

    if(server->pub_ip) {
        DEBUG_OUT("public IP is %s\n", server->pub_ip);
    }

    if(server->priv_ip) {
        DEBUG_OUT("private IP is %s\n", server->priv_ip);
    }

    *sv = server;

    is_initialized = 1;

    DEBUG_OUT("server is ready for requests.\n");

    return dspaces_SUCCESS;
}

static void kill_client(dspaces_provider_t server, char *client_addr)
{
    hg_addr_t server_addr;
    hg_handle_t h;
    margo_request req;
    int arg = -1;

    margo_addr_lookup(server->mid, client_addr, &server_addr);
    margo_create(server->mid, server_addr, server->kill_client_id, &h);
    margo_iforward(h, &arg, &req);
    margo_addr_free(server->mid, server_addr);
    margo_destroy(h);
}

/*
 * Clients with local data need to know when it's safe to finalize. Send kill
 * rpc to any clients in the drain list.
 */
static void kill_local_clients(dspaces_provider_t server)
{
    struct obj_desc_list *odscl;
    struct list_head client_list;
    struct addr_list_entry *client_addr, *temp;
    int found;

    INIT_LIST_HEAD(&client_list);

    DEBUG_OUT("Killing clients with local storage.\n");

    ABT_mutex_lock(server->odsc_mutex);
    list_for_each_entry(odscl, &(server->dsg->obj_desc_drain_list),
                        struct obj_desc_list, odsc_entry)
    {
        found = 0;
        list_for_each_entry(client_addr, &client_list, struct addr_list_entry,
                            entry)
        {
            if(strcmp(client_addr->addr, odscl->odsc.owner) == 0) {
                found = 1;
                break;
            }
        }
        if(!found) {
            DEBUG_OUT("Adding %s to kill list.\n", odscl->odsc.owner);
            client_addr = malloc(sizeof(*client_addr));
            client_addr->addr = strdup(odscl->odsc.owner);
            list_add(&client_addr->entry, &client_list);
        }
    }
    ABT_mutex_unlock(server->odsc_mutex);

    list_for_each_entry_safe(client_addr, temp, &client_list,
                             struct addr_list_entry, entry)
    {
        DEBUG_OUT("Sending kill signal to %s.\n", client_addr->addr);
        kill_client(server, client_addr->addr);
        list_del(&client_addr->entry);
        free(client_addr->addr);
        free(client_addr);
    }
}

static int server_destroy(dspaces_provider_t server)
{
    int i;
    MPI_Barrier(server->comm);
    DEBUG_OUT("Finishing up, waiting for asynchronous jobs to finish...\n");

    if(server->f_drain) {
        ABT_thread_free(&server->drain_t);
        ABT_xstream_join(server->drain_xstream);
        ABT_xstream_free(&server->drain_xstream);
        DEBUG_OUT("drain thread stopped.\n");
    }

    kill_local_clients(server);

    // Hack to avoid possible argobots race condition. Need to track this down
    // at some point.
    sleep(5);

    for(i = 0; i < server->nremote; i++) {
        dspaces_fini(server->remotes[i].conn);
    }

    free_sspace(server->dsg);
    ls_free(server->dsg->ls);
    free(server->dsg);
    free(server->server_address[0]);
    free(server->server_address);
    free(server->listen_addr_str);

    MPI_Barrier(server->comm);
    MPI_Comm_free(&server->comm);
    DEBUG_OUT("finalizing server.\n");
    margo_finalize(server->mid);
    DEBUG_OUT("finalized server.\n");
    return 0;
}

static void address_translate(dspaces_provider_t server, char *addr_str)
{
    char *addr_loc = strstr(addr_str, server->priv_ip);
    char *addr_tail;
    int publen, privlen;

    if(addr_loc) {
        DEBUG_OUT("translating %s.\n", addr_str);
        publen = strlen(server->pub_ip);
        privlen = strlen(server->priv_ip);
        addr_tail = strdup(addr_loc + privlen);
        strcpy(addr_loc, server->pub_ip);
        strcat(addr_str, addr_tail);
        free(addr_tail);
        DEBUG_OUT("translated address: %s\n", addr_str);
    } else {
        DEBUG_OUT("no translation needed.\n");
    }
}

static void odsc_take_ownership(dspaces_provider_t server, obj_descriptor *odsc)
{
    hg_addr_t owner_addr;
    hg_size_t owner_addr_size = 128;

    margo_addr_self(server->mid, &owner_addr);
    margo_addr_to_string(server->mid, odsc->owner, &owner_addr_size,
                         owner_addr);
    if(server->pub_ip && server->priv_ip) {
        address_translate(server, odsc->owner);
    }
}

static void put_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    bulk_gdim_t in;
    bulk_out_t out;
    hg_bulk_t bulk_handle;
    struct timeval start, stop;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    if(server->f_kill == 0) {
        fprintf(stderr, "WARNING: put rpc received when server is finalizing. "
                        "This will likely cause problems...\n");
    }

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));
    // set the owner to be this server address
    odsc_take_ownership(server, &in_odsc);

    struct obj_data *od;
    od = obj_data_alloc(&in_odsc);
    memcpy(&od->gdim, in.odsc.raw_gdim, sizeof(struct global_dimension));

    if(!od)
        fprintf(stderr, "ERROR: (%s): object allocation failed!\n", __func__);

    // do write lock

    hg_size_t size = (in_odsc.size) * bbox_volume(&(in_odsc.bb));

    DEBUG_OUT("Creating a bulk transfer buffer of size %" PRIu64 "\n", size);

    hret = margo_bulk_create(mid, 1, (void **)&(od->data), &size,
                             HG_BULK_WRITE_ONLY, &bulk_handle);

    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create failed!\n", __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        return;
    }

    gettimeofday(&start, NULL);

    hret = margo_bulk_transfer(mid, HG_BULK_PULL, info->addr, in.handle, 0,
                               bulk_handle, 0, size);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_transfer failed!\n", __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_bulk_free(bulk_handle);
        margo_destroy(handle);
        return;
    }

    gettimeofday(&stop, NULL);

    if(server->f_debug) {
        long dsec = stop.tv_sec - start.tv_sec;
        long dusec = stop.tv_usec - start.tv_usec;
        float transfer_time = (float)dsec + (dusec / 1000000.0);
        DEBUG_OUT("got %" PRIu64 " bytes in %f sec\n", size, transfer_time);
    }

    ABT_mutex_lock(server->ls_mutex);
    ls_add_obj(server->dsg->ls, od);
    ABT_mutex_unlock(server->ls_mutex);

    DEBUG_OUT("Received obj %s\n", obj_desc_sprint(&od->obj_desc));

    // now update the dht
    out.ret = dspaces_SUCCESS;
    margo_bulk_free(bulk_handle);
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);

    obj_update_dht(server, od, DS_OBJ_NEW);
    DEBUG_OUT("Finished obj_put_update from put_rpc\n");
}
DEFINE_MARGO_RPC_HANDLER(put_rpc)

static void put_local_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    odsc_gdim_t in;
    bulk_out_t out;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("In the local put rpc\n");

    if(server->f_kill == 0) {
        fprintf(stderr,
                "WARNING: (%s): got put rpc with local storage, but server is "
                "shutting down. This will likely cause problems...\n",
                __func__);
    }

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc_gdim.raw_odsc, sizeof(in_odsc));

    struct obj_data *od;
    od = obj_data_alloc_no_data(&in_odsc, NULL);
    memcpy(&od->gdim, in.odsc_gdim.raw_gdim, sizeof(struct global_dimension));

    if(!od)
        fprintf(stderr, "ERROR: (%s): failed to allocate object data!\n",
                __func__);

    DEBUG_OUT("Received obj %s  in put_local_rpc\n",
              obj_desc_sprint(&od->obj_desc));

    // now update the dht
    obj_update_dht(server, od, DS_OBJ_NEW);
    DEBUG_OUT("Finished obj_put_local_update in local_put\n");

    // add to the local list for marking as to be drained data
    struct obj_desc_list *odscl;
    odscl = malloc(sizeof(*odscl));
    memcpy(&odscl->odsc, &od->obj_desc, sizeof(obj_descriptor));

    ABT_mutex_lock(server->odsc_mutex);
    DEBUG_OUT("Adding drain list entry.\n");
    list_add_tail(&odscl->odsc_entry, &server->dsg->obj_desc_drain_list);
    ABT_mutex_unlock(server->odsc_mutex);

    // TODO: wake up thread to initiate draining
    out.ret = dspaces_SUCCESS;
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);

    free(od);
}
DEFINE_MARGO_RPC_HANDLER(put_local_rpc)

static void put_meta_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    hg_return_t hret;
    put_meta_in_t in;
    bulk_out_t out;
    struct meta_data *mdata;
    hg_size_t rdma_size;
    hg_bulk_t bulk_handle;

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    DEBUG_OUT("Received meta data of length %d, name '%s' version %d.\n",
              in.length, in.name, in.version);

    mdata = malloc(sizeof(*mdata));
    mdata->name = strdup(in.name);
    mdata->version = in.version;
    mdata->length = in.length;
    mdata->data = (in.length > 0) ? malloc(in.length) : NULL;

    rdma_size = mdata->length;
    if(rdma_size > 0) {
        hret = margo_bulk_create(mid, 1, (void **)&mdata->data, &rdma_size,
                                 HG_BULK_WRITE_ONLY, &bulk_handle);

        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_create failed!\n",
                    __func__);
            out.ret = dspaces_ERR_MERCURY;
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_bulk_free(bulk_handle);
            margo_destroy(handle);
            return;
        }

        hret = margo_bulk_transfer(mid, HG_BULK_PULL, info->addr, in.handle, 0,
                                   bulk_handle, 0, rdma_size);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_transfer failed!\n",
                    __func__);
            out.ret = dspaces_ERR_MERCURY;
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_bulk_free(bulk_handle);
            margo_destroy(handle);
            return;
        }
    }

    DEBUG_OUT("adding to metaddata store.\n");

    ABT_mutex_lock(server->ls_mutex);
    ls_add_meta(server->dsg->ls, mdata);
    ABT_mutex_unlock(server->ls_mutex);

    DEBUG_OUT("successfully stored.\n");

    out.ret = dspaces_SUCCESS;
    if(rdma_size > 0) {
        margo_bulk_free(bulk_handle);
    }
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(put_meta_rpc)

static int query_remotes(dspaces_provider_t server, obj_descriptor *q_odsc,
                         struct global_dimension *q_gdim, int timeout,
                         obj_descriptor **results, int req_id)
{
    uint64_t buf_size;
    void *buffer = NULL;
    int found = 0;
    obj_descriptor *odsc;
    struct obj_data *od;
    hg_addr_t owner_addr;
    hg_size_t owner_addr_size = 128;
    int i;

    buf_size = obj_data_size(q_odsc);
    buffer = malloc(buf_size);

    DEBUG_OUT("%i remotes to query\n", server->nremote);
    for(i = 0; i < server->nremote; i++) {
        DEBUG_OUT("req %i: querying remote %s\n", req_id,
                  server->remotes[i].name);
        dspaces_define_gdim(server->remotes[i].conn, q_odsc->name, q_gdim->ndim,
                            q_gdim->sizes.c);
        if(dspaces_get(server->remotes[i].conn, q_odsc->name, q_odsc->version,
                       q_odsc->size, q_odsc->bb.num_dims, q_odsc->bb.lb.c,
                       q_odsc->bb.ub.c, buffer, 0) == 0) {
            DEBUG_OUT("found data\n");
            found = 1;
            break;
        }
    }

    if(found) {
        odsc = calloc(1, sizeof(*odsc));
        odsc->version = q_odsc->version;
        margo_addr_self(server->mid, &owner_addr);
        margo_addr_to_string(server->mid, odsc->owner, &owner_addr_size,
                             owner_addr);
        margo_addr_free(server->mid, owner_addr);
        // Selectively translate address?
        odsc->st = q_odsc->st;
        odsc->size = q_odsc->size;
        memcpy(&odsc->bb, &q_odsc->bb, sizeof(odsc->bb));
        od = obj_data_alloc_no_data(odsc, buffer);
        *results = odsc;
        if(server->f_debug) {
            DEBUG_OUT("created local object %s\n", obj_desc_sprint(odsc));
        }
        ABT_mutex_lock(server->ls_mutex);
        ls_add_obj(server->dsg->ps, od);
        ABT_mutex_unlock(server->ls_mutex);
        return (sizeof(obj_descriptor));
    } else {
        DEBUG_OUT("req %i: not found on any remotes.\n", req_id);
        *results = NULL;
        return (0);
    }
}

static int route_request(dspaces_provider_t server, obj_descriptor *odsc,
                         struct global_dimension *gdim)
{
    struct dspaces_module *mod;
    struct dspaces_module_args *args;
    struct dspaces_module_ret *res = NULL;
    struct obj_data *od;
    obj_descriptor *od_odsc;
    int nargs;
    int i, err;

    DEBUG_OUT("Routing '%s'\n", odsc->name);

    mod = dspaces_mod_by_od(&server->mods, odsc);
    if(mod) {
        nargs = build_module_args_from_odsc(odsc, &args);
        res = dspaces_module_exec(mod, "query", args, nargs,
                                  DSPACES_MOD_RET_ARRAY);
    }

    if(res && res->type == DSPACES_MOD_RET_ERR) {
        err = res->err;
        free(res);
        return (err);
    }

    if(res) {
        if(odsc->size && odsc->size != res->elem_size) {
            fprintf(stderr,
                    "WARNING: user requested data with element size %zi, but "
                    "module routing resulted in element size %i. Not adding "
                    "anything to local storage.\n",
                    odsc->size, res->elem_size);
            free(res->data);
        } else {
            odsc->size = res->elem_size;
            if(odsc->bb.num_dims != res->ndim) {
                DEBUG_OUT("change in dimensionality.\n");
                odsc->bb.num_dims = res->ndim;
                obj_data_resize(odsc, res->dim);
            } else if((odsc->size * res->len) != obj_data_size(odsc)) {
                DEBUG_OUT("returned data is cropped.\n");
                obj_data_resize(odsc, res->dim);
            }
            // TODO: refactor to convert ret->odsc in function
            od_odsc = malloc(sizeof(*od_odsc));
            memcpy(od_odsc, odsc, sizeof(*od_odsc));
            odsc_take_ownership(server, od_odsc);
            od_odsc->tag = res->tag;
            if(odsc->tag == 0) {
                odsc->tag = res->tag;
            }
            od = obj_data_alloc_no_data(od_odsc, res->data);
            memcpy(&od->gdim, gdim, sizeof(struct global_dimension));
            DEBUG_OUT("adding object to local storage: %s\n",
                      obj_desc_sprint(od_odsc));
            ABT_mutex_lock(server->ls_mutex);
            ls_add_obj(server->dsg->ls, od);
            ABT_mutex_unlock(server->ls_mutex);

            obj_update_dht(server, od, DS_OBJ_NEW);
        }
        free(res);
    }

    return (0);
}

// should we handle this locally
static int local_responsibility(dspaces_provider_t server, obj_descriptor *odsc)
{
    return (!server->remotes || ls_lookup(server->dsg->ls, odsc->name));
}

static void send_tgt_rpc(dspaces_provider_t server, hg_id_t rpc_id, int target,
                         void *in, hg_addr_t *addr, hg_handle_t *h,
                         margo_request *req)
{
    DEBUG_OUT("sending rpc_id %" PRIu64 " to rank %i\n", rpc_id, target);
    margo_addr_lookup(server->mid, server->server_address[target], addr);
    margo_create(server->mid, *addr, rpc_id, h);
    margo_iforward(*h, in, req);
}

struct ibcast_state {
    int sent_rpc[3];
    hg_addr_t addr[3];
    hg_handle_t hndl[3];
    margo_request req[3];
};

static struct ibcast_state *ibcast_rpc_start(dspaces_provider_t server,
                                             hg_id_t rpc_id, int32_t src,
                                             void *in)
{
    struct ibcast_state *bcast;
    int32_t rank, parent, child1, child2;

    rank = server->dsg->rank;
    parent = (rank - 1) / 2;
    child1 = (rank * 2) + 1;
    child2 = child1 + 1;

    if(server->dsg->size_sp == 1 ||
       (src == parent && child1 >= server->dsg->size_sp)) {
        return NULL;
    }

    bcast = calloc(1, sizeof(*bcast));

    if((src == -1 || src > rank) && rank > 0) {
        DEBUG_OUT("sending to parent (%i)\n", parent);
        send_tgt_rpc(server, rpc_id, parent, in, &bcast->addr[0],
                     &bcast->hndl[0], &bcast->req[0]);
        bcast->sent_rpc[0] = 1;
    }
    if((child1 != src && child1 < server->dsg->size_sp)) {
        DEBUG_OUT("sending to child 1 (%i)\n", child1);
        send_tgt_rpc(server, rpc_id, child1, in, &bcast->addr[1],
                     &bcast->hndl[1], &bcast->req[1]);
        bcast->sent_rpc[1] = 1;
    }
    if((child2 != src && child2 < server->dsg->size_sp)) {
        DEBUG_OUT("sending to child 2 (%i)\n", child2);
        send_tgt_rpc(server, rpc_id, child2, in, &bcast->addr[2],
                     &bcast->hndl[2], &bcast->req[2]);
        bcast->sent_rpc[2] = 1;
    }

    return (bcast);
}

static void ibcast_get_output(struct ibcast_state *bcast, unsigned int idx,
                              void *out, int *present)
{
    if(idx > 2 || !bcast->sent_rpc[idx]) {
        if(present) {
            *present = 0;
        }
        return;
    }

    margo_wait(bcast->req[idx]);
    margo_get_output(bcast->hndl[idx], out);
    if(present) {
        *present = 1;
    }
}

static void ibcast_finish(dspaces_provider_t server, struct ibcast_state *bcast)
{
    int i;

    for(i = 0; i < 3; i++) {
        if(bcast->sent_rpc[i]) {
            margo_addr_free(server->mid, bcast->addr[i]);
            margo_destroy(bcast->hndl[i]);
        }
    }
    free(bcast);
}

static int get_query_odscs(dspaces_provider_t server, odsc_gdim_t *query,
                           int timeout, obj_descriptor **results, int req_id)
{
    struct sspace *ssd;
    struct dht_entry **de_tab;
    int peer_num;
    int self_id_num = -1;
    int total_odscs = 0;
    int *odsc_nums;
    obj_descriptor **odsc_tabs, **podsc = NULL;
    obj_descriptor *odsc_curr;
    margo_request *serv_reqs;
    hg_handle_t *hndls;
    hg_addr_t server_addr;
    odsc_list_t dht_resp;
    obj_descriptor *q_odsc;
    struct global_dimension *q_gdim;
    int dup;
    int i, j, k;

    q_odsc = (obj_descriptor *)query->odsc_gdim.raw_odsc;
    q_gdim = (struct global_dimension *)query->odsc_gdim.raw_gdim;

    if(!local_responsibility(server, q_odsc)) {
        DEBUG_OUT("req %i: no local objects with name %s. Checking remotes.\n",
                  req_id, q_odsc->name);
        return (
            query_remotes(server, (obj_descriptor *)query->odsc_gdim.raw_odsc,
                          (struct global_dimension *)query->odsc_gdim.raw_gdim,
                          timeout, results, req_id));
    }

    DEBUG_OUT("getting sspace lock.\n");
    ABT_mutex_lock(server->sspace_mutex);
    DEBUG_OUT("got lock, looking up shared space for global dimensions.\n");
    ssd = lookup_sspace(server, q_odsc->name, q_gdim);
    ABT_mutex_unlock(server->sspace_mutex);
    DEBUG_OUT("found shared space with %i entries.\n", ssd->dht->num_entries);

    de_tab = malloc(sizeof(*de_tab) * ssd->dht->num_entries);
    peer_num = ssd_hash(ssd, &(q_odsc->bb), de_tab);

    DEBUG_OUT("%d peers to query\n", peer_num);

    odsc_tabs = malloc(sizeof(*odsc_tabs) * peer_num);
    odsc_nums = calloc(sizeof(*odsc_nums), peer_num);
    serv_reqs = malloc(sizeof(*serv_reqs) * peer_num);
    hndls = malloc(sizeof(*hndls) * peer_num);

    for(i = 0; i < peer_num; i++) {
        DEBUG_OUT("dht server id %d\n", de_tab[i]->rank);
        DEBUG_OUT("self id %d\n", server->dsg->rank);

        if(de_tab[i]->rank == server->dsg->rank) {
            self_id_num = i;
            continue;
        }
        // remote servers
        margo_addr_lookup(server->mid, server->server_address[de_tab[i]->rank],
                          &server_addr);
        margo_create(server->mid, server_addr, server->odsc_internal_id,
                     &hndls[i]);
        margo_iforward(hndls[i], query, &serv_reqs[i]);
        margo_addr_free(server->mid, server_addr);
    }

    if(peer_num == 0) {
        DEBUG_OUT("no peers in global space, handling with modules only\n");
        odsc_tabs = malloc(sizeof(*odsc_tabs));
        odsc_nums = calloc(sizeof(*odsc_nums), 1);
        self_id_num = 0;
    }

    if(self_id_num > -1) {
        route_request(server, q_odsc, q_gdim);
        DEBUG_OUT("finding local entries for req_id %i.\n", req_id);
        odsc_nums[self_id_num] =
            dht_find_entry_all(ssd->ent_self, q_odsc, &podsc, timeout);
        DEBUG_OUT("%d odscs found in %d\n", odsc_nums[self_id_num],
                  server->dsg->rank);
        total_odscs += odsc_nums[self_id_num];
        if(odsc_nums[self_id_num]) {
            odsc_tabs[self_id_num] =
                malloc(sizeof(**odsc_tabs) * odsc_nums[self_id_num]);
            for(i = 0; i < odsc_nums[self_id_num]; i++) {
                obj_descriptor *odsc =
                    &odsc_tabs[self_id_num][i]; // readability
                *odsc = *podsc[i];
                odsc->st = q_odsc->st;
                bbox_intersect(&q_odsc->bb, &odsc->bb, &odsc->bb);
                DEBUG_OUT("%s\n", obj_desc_sprint(&odsc_tabs[self_id_num][i]));
            }
        }

        free(podsc);
    }

    for(i = 0; i < peer_num; i++) {
        if(i == self_id_num) {
            continue;
        }
        DEBUG_OUT("req_id %i waiting for %d\n", req_id, i);
        margo_wait(serv_reqs[i]);
        margo_get_output(hndls[i], &dht_resp);
        if(dht_resp.odsc_list.size != 0) {
            odsc_nums[i] = dht_resp.odsc_list.size / sizeof(obj_descriptor);
            DEBUG_OUT("received %d odscs from peer %d for req_id %i\n",
                      odsc_nums[i], i, req_id);
            total_odscs += odsc_nums[i];
            odsc_tabs[i] = malloc(sizeof(**odsc_tabs) * odsc_nums[i]);
            memcpy(odsc_tabs[i], dht_resp.odsc_list.raw_odsc,
                   dht_resp.odsc_list.size);

            for(j = 0; j < odsc_nums[i]; j++) {
                // readability
                obj_descriptor *odsc =
                    (obj_descriptor *)dht_resp.odsc_list.raw_odsc;
                DEBUG_OUT("remote buffer: %s\n", obj_desc_sprint(&odsc[j]));
            }
        }
        margo_free_output(hndls[i], &dht_resp);
        margo_destroy(hndls[i]);
    }

    odsc_curr = *results = malloc(sizeof(**results) * total_odscs);

    if(peer_num == 0)
        peer_num = 1;
    for(i = 0; i < peer_num; i++) {
        if(odsc_nums[i] == 0) {
            continue;
        }
        // dedup
        for(j = 0; j < odsc_nums[i]; j++) {
            dup = 0;
            for(k = 0; k < (odsc_curr - *results); k++) {
                if(obj_desc_equals_no_owner(&(*results)[k], &odsc_tabs[i][j])) {
                    dup = 1;
                    total_odscs--;
                    break;
                }
            }
            if(!dup) {
                *odsc_curr = odsc_tabs[i][j];
                odsc_curr++;
            }
        }
        free(odsc_tabs[i]);
    }

    for(i = 0; i < total_odscs; i++) {
        DEBUG_OUT("odsc %i in response for req_id %i: %s\n", i, req_id,
                  obj_desc_sprint(&(*results)[i]));
    }

    free(de_tab);
    free(hndls);
    free(serv_reqs);
    free(odsc_tabs);
    free(odsc_nums);

    return (sizeof(obj_descriptor) * total_odscs);
}

static void query_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    dspaces_provider_t server;
    odsc_gdim_t in;
    odsc_list_t out;
    obj_descriptor in_odsc;
    struct global_dimension in_gdim;
    int timeout;
    obj_descriptor *results;
    hg_return_t hret;
    static int uid = 0;
    int req_id;

    req_id = __sync_fetch_and_add(&uid, 1);

    // unwrap context and input from margo
    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    server = (dspaces_provider_t)margo_registered_data(mid, info->id);
    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    DEBUG_OUT("received query\n");

    memcpy(&in_odsc, in.odsc_gdim.raw_odsc, sizeof(in_odsc));
    memcpy(&in_gdim, in.odsc_gdim.raw_gdim, sizeof(struct global_dimension));
    timeout = in.param;
    DEBUG_OUT("Received query for %s with timeout %d\n",
              obj_desc_sprint(&in_odsc), timeout);

    out.odsc_list.size =
        get_query_odscs(server, &in, timeout, &results, req_id);

    out.odsc_list.raw_odsc = (char *)results;
    DEBUG_OUT("Responding with %li result(s).\n",
              out.odsc_list.size / sizeof(obj_descriptor));
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);

    free(results);
}
DEFINE_MARGO_RPC_HANDLER(query_rpc)

static int param_hash(int type, void *val, int mod)
{
    char *sval;
    int hval = 1;
    int i;

    switch(type) {
    case DSP_BOOL:
    case DSP_CHAR:
    case DSP_BYTE:
    case DSP_UINT8:
    case DSP_INT8:
        return (*(uint8_t *)val);
    case DSP_UINT16:
    case DSP_INT16:
        return (*(uint16_t *)val);
    case DSP_FLOAT:
    case DSP_INT:
    case DSP_UINT:
    case DSP_UINT32:
    case DSP_INT32:
        return (*(uint32_t *)val);
    case DSP_LONG:
    case DSP_DOUBLE:
    case DSP_ULONG:
    case DSP_UINT64:
    case DSP_INT64:
        return ((*(uint64_t *)val) % mod);
    case DSP_STR:
    case DSP_JSON:
        sval = val;
        for(i = 0; sval[i]; i++) {
            hval = (hval * sval[i]) % mod;
        }
        return (hval);
    }
}

static unsigned int params_hash(dsp_dict_t *params)
{
    long hval = 1;
    static const int mod = 1000000007;
    int i;

    for(i = 0; i < params->len; i++) {
        hval =
            (hval * param_hash(params->types[i], params->vals[i], mod)) % mod;
    }

    return (hval);
}

static void write_param_name(dsp_dict_t *params, char name[OD_MAX_NAME_LEN])
{
    char *pstr_start = name;
    static const int mod = 1000000007;
    int hval;
    int i;

    for(i = 0; i < params->len; i++) {
        hval = param_hash(params->types[i], params->vals[i], mod);
        sprintf(pstr_start, "%x.4", hval);
        pstr_start += 8;
        if(pstr_start - name >= 136) {
            pstr_start = name;
        }
    }
}

static void get_version_name(dsp_dict_t *params, unsigned int *version,
                             char name[OD_MAX_NAME_LEN])
{
    int ver_set = 0;
    int name_set = 0;
    int i;

    for(i = 0; i < params->len; i++) {
        if(params->types[i] == DSP_STR &&
           strcmp(params->keys[i], "name") == 0) {
            memcpy(name, params->vals[i], 149);
            name_set = 1;
        }
        if(params->types[i] == DSP_UINT &&
           strcmp(params->keys[i], "version") == 0) {
            *version = *(unsigned int *)(params->vals[i]);
            ver_set = 1;
        }
    }
    if(!ver_set) {
        *version = params_hash(params);
    }
    if(!name_set) {
        write_param_name(params, name);
    }
}

static void get_mod_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    get_mod_in_t in;
    get_mod_out_t out = {0};
    struct dspaces_module *mod;
    struct dspaces_module_args *args;
    int nargs;
    struct dspaces_module_ret *res = NULL;
    hg_return_t hret;
    char name[OD_MAX_NAME_LEN] = {0};
    unsigned int version;
    obj_descriptor *odsc;
    struct obj_data *od;
    int8_t *types;
    int i, err;

    HG_TRY(margo_get_input(handle, &in), 0, err_destroy,
           "margo_get_input() failed.\n");

    mod = dspaces_mod_by_name(&server->mods, in.name);
    if(!mod) {
        fprintf(stderr, "ERROR: %s: module '%s' requested but not available.\n",
                __func__, in.name);
        out.ret = DS_MOD_ENOMOD;
        goto err_respond;
    }

    // To keep modules from being depedent on hg types.
    // vals are void * ; we can probably depend on hg_string_t
    // being a typedef of char *, and there's no hg_string_t
    // conversion interface anyway...
    types = malloc(sizeof(*types) * in.params.len);
    for(i = 0; i < in.params.len; i++) {
        types[i] = in.params.types[i];
    }

    nargs = build_module_args_from_dict(in.params.len, types, in.params.keys,
                                        in.params.vals, &args);
    free(types);
    res =
        dspaces_module_exec(mod, "pquery", args, nargs, DSPACES_MOD_RET_ARRAY);
    if(res && res->type == DSPACES_MOD_RET_ERR) {
        DEBUG_OUT("Module failure. Returning EFAULT to user.\n")
        out.ret = DS_MOD_EFAULT;
        free(res);
        goto err_respond;
    } else if(res) {
        get_version_name(&in.params, &version, name);
    }

    odsc_from_ret(res, &odsc, name, version);
    odsc_take_ownership(server, odsc);
    od = obj_data_alloc_no_data(odsc, res->data);
    ABT_mutex_lock(server->ls_mutex);
    ls_add_obj(server->dsg->ls, od);
    ABT_mutex_unlock(server->ls_mutex);
    obj_update_dht(server, od, DS_OBJ_NEW);

    out.odscs.odsc_list.raw_odsc = (char *)odsc;
    out.odscs.odsc_list.size = sizeof(*odsc);

    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);

    free(odsc);

    return;

err_respond:
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
err_destroy:
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(get_mod_rpc)

static int peek_meta_remotes(dspaces_provider_t server, peek_meta_in_t *in)
{
    margo_request *reqs;
    hg_handle_t *peek_hndls;
    peek_meta_out_t *resps;
    hg_addr_t addr;
    int i;
    int ret = -1;
    size_t index;
    hg_return_t hret;

    reqs = malloc(sizeof(*reqs) * server->nremote);
    resps = malloc(sizeof(*resps) * server->nremote);
    peek_hndls = malloc(sizeof(*peek_hndls) * server->nremote);

    DEBUG_OUT("sending peek request to remotes for metadata '%s'\n", in->name);
    for(i = 0; i < server->nremote; i++) {
        DEBUG_OUT("querying %s at %s\n", server->remotes[i].name,
                  server->remotes[i].addr_str);
        margo_addr_lookup(server->mid, server->remotes[i].addr_str, &addr);
        hret = margo_create(server->mid, addr, server->peek_meta_id,
                            &peek_hndls[i]);
        hret = margo_iforward(peek_hndls[i], in, &reqs[i]);
        margo_addr_free(server->mid, addr);
    }

    for(i = 0; i < server->nremote; i++) {
        hret = margo_wait_any(server->nremote, reqs, &index);
        margo_get_output(peek_hndls[index], &resps[index]);
        DEBUG_OUT("%s replied with %i\n", server->remotes[index].name,
                  resps[index].res);
        if(resps[index].res == 1) {
            ret = i;
        }
        margo_free_output(peek_hndls[index], &resps[index]);
        margo_destroy(peek_hndls[index]);
    }

    free(reqs);
    free(resps);
    free(peek_hndls);

    return (ret);
}

static void peek_meta_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    peek_meta_in_t in;
    peek_meta_out_t out;
    hg_return_t hret;
    hg_addr_t addr;

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    DEBUG_OUT("received peek request for metadata '%s'\n", in.name);

    out.res = 0;

    if(meta_find_next_entry(server->dsg->ls, in.name, -1, 0)) {
        DEBUG_OUT("found the metadata\n");
        out.res = 1;
    } else if(server->nremote) {
        DEBUG_OUT("no such metadata in local storage.\n");
        if(peek_meta_remotes(server, &in) > -1) {
            out.res = 1;
        }
    }

    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(peek_meta_rpc)

static void query_meta_rpc(hg_handle_t handle)
{
    margo_instance_id mid;
    const struct hg_info *info;
    dspaces_provider_t server;
    query_meta_in_t in;
    query_meta_out_t rem_out, out;
    peek_meta_in_t rem_in;
    struct meta_data *mdata, *mdlatest;
    int remote, found_remote;
    hg_handle_t rem_hndl;
    hg_addr_t rem_addr;
    hg_return_t hret;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    server = (dspaces_provider_t)margo_registered_data(mid, info->id);
    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    DEBUG_OUT("received metadata query for version %d of '%s', mode %d.\n",
              in.version, in.name, in.mode);

    found_remote = 0;
    if(server->nremote) {
        rem_in.name = in.name;
        remote = peek_meta_remotes(server, &rem_in);
        if(remote > -1) {
            DEBUG_OUT("remote %s has %s metadata\n",
                      server->remotes[remote].name, in.name);
            margo_addr_lookup(server->mid, server->remotes[remote].addr_str,
                              &rem_addr);
            hret = margo_create(server->mid, rem_addr, server->query_meta_id,
                                &rem_hndl);
            if(hret != HG_SUCCESS) {
                fprintf(stderr, "ERROR: (%s): margo_create() failed\n",
                        __func__);
                margo_addr_free(server->mid, rem_addr);
            } else {
                margo_forward(rem_hndl, &in);
                hret = margo_get_output(rem_hndl, &out);
                if(hret != HG_SUCCESS) {
                    fprintf(stderr,
                            "ERROR: %s: margo_get_output() failed with %d.\n",
                            __func__, hret);
                } else {
                    DEBUG_OUT("retreived metadata from %s\n",
                              server->remotes[remote].name);
                    found_remote = 1;
                }
            }
        }
    }
    if(!found_remote) {
        switch(in.mode) {
        case META_MODE_SPEC:
            DEBUG_OUT("spec query - searching without waiting...\n");
            mdata = meta_find_entry(server->dsg->ls, in.name, in.version, 0);
            break;
        case META_MODE_NEXT:
            DEBUG_OUT("find next query...\n");
            mdata =
                meta_find_next_entry(server->dsg->ls, in.name, in.version, 1);
            break;
        case META_MODE_LAST:
            DEBUG_OUT("find last query...\n");
            mdata =
                meta_find_next_entry(server->dsg->ls, in.name, in.version, 1);
            mdlatest = mdata;
            do {
                mdata = mdlatest;
                DEBUG_OUT("found version %d. Checking for newer...\n",
                          mdata->version);
                mdlatest = meta_find_next_entry(server->dsg->ls, in.name,
                                                mdlatest->version, 0);
            } while(mdlatest);
            break;
        default:
            fprintf(stderr,
                    "ERROR: unkown mode %d while processing metadata query.\n",
                    in.mode);
        }

        if(mdata) {
            DEBUG_OUT("found version %d, length %d.", mdata->version,
                      mdata->length);
            out.mdata.len = mdata->length;
            out.mdata.buf = malloc(mdata->length);
            memcpy(out.mdata.buf, mdata->data, mdata->length);
            out.version = mdata->version;
        } else {
            out.mdata.len = 0;
            out.version = -1;
        }
    }
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    if(found_remote) {
        margo_free_output(rem_hndl, &out);
        margo_destroy(rem_hndl);
    }
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(query_meta_rpc)

static void get_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    bulk_in_t in;
    bulk_out_t out;
    hg_bulk_t bulk_handle;
    int csize;
    int suppress_compression;
    void *cbuffer;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    suppress_compression = in.flags & DS_NO_COMPRESS;

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));

    DEBUG_OUT("received get request\n");

    struct obj_data *od, *from_obj;

    ABT_mutex_lock(server->ls_mutex);
    if(server->remotes && ls_lookup(server->dsg->ps, in_odsc.name)) {
        from_obj = ls_find(server->dsg->ps, &in_odsc);
    } else {
        from_obj = ls_find(server->dsg->ls, &in_odsc);
    }
    DEBUG_OUT("found source data object\n");
    od = obj_data_alloc(&in_odsc);
    DEBUG_OUT("allocated target object\n");
    ssd_copy(od, from_obj);
    DEBUG_OUT("copied object data\n");
    ABT_mutex_unlock(server->ls_mutex);
    hg_size_t size = (in_odsc.size) * bbox_volume(&(in_odsc.bb));
    void *buffer = (void *)od->data;
    cbuffer = malloc(size);
    hret = margo_bulk_create(mid, 1, (void **)&cbuffer, &size,
                             HG_BULK_READ_ONLY, &bulk_handle);
    DEBUG_OUT("created bulk handle of size %" PRIu64 "\n", size);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failure\n", __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        return;
    }

    csize = 0;
    if(!suppress_compression) {
        /* things like CPU->GPU transfer don't support compression, but we need
         * the client to tell us. */
        csize = LZ4_compress_default(od->data, cbuffer, size, size);
        DEBUG_OUT("compressed result from %" PRIu64 " to %i bytes.\n", size,
                  csize);
        if(!csize) {
            DEBUG_OUT(
                "compressed result could not fit in dst buffer - longer than "
                "original! Sending uncompressed.\n");
        }
    } else {
        DEBUG_OUT("Compression suppression flag set. Skipping lz4.\n");
    }

    if(!csize) {
        memcpy(cbuffer, od->data, size);
    }

    hret = margo_bulk_transfer(mid, HG_BULK_PUSH, info->addr, in.handle, 0,
                               bulk_handle, 0, (csize ? csize : size));
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_transfer() failure (%d)\n",
                __func__, hret);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_bulk_free(bulk_handle);
        margo_destroy(handle);
        return;
    }
    DEBUG_OUT("completed bulk transfer.\n");
    margo_bulk_free(bulk_handle);
    out.ret = dspaces_SUCCESS;
    out.len = csize;
    obj_data_free(od);
    margo_respond(handle, &out);
    margo_free_input(handle, &in);
    margo_destroy(handle);
    free(cbuffer);
}
DEFINE_MARGO_RPC_HANDLER(get_rpc)

static void odsc_internal_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    odsc_gdim_t in;
    int timeout;
    odsc_list_t out;
    obj_descriptor **podsc = NULL;
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    margo_request req;

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc_gdim.raw_odsc, sizeof(in_odsc));
    timeout = in.param;

    struct global_dimension od_gdim;
    memcpy(&od_gdim, in.odsc_gdim.raw_gdim, sizeof(struct global_dimension));

    DEBUG_OUT("Received query for %s with timeout %d\n",
              obj_desc_sprint(&in_odsc), timeout);

    obj_descriptor *odsc_tab;
    DEBUG_OUT("getting sspace lock.\n");
    ABT_mutex_lock(server->sspace_mutex);
    DEBUG_OUT("got sspace lock.\n");
    struct sspace *ssd = lookup_sspace(server, in_odsc.name, &od_gdim);
    ABT_mutex_unlock(server->sspace_mutex);
    route_request(server, &in_odsc, &od_gdim);
    int num_odsc;
    num_odsc = dht_find_entry_all(ssd->ent_self, &in_odsc, &podsc, timeout);
    DEBUG_OUT("found %d DHT entries.\n", num_odsc);
    if(!num_odsc) {
        // need to figure out how to send that number of odscs is null
        out.odsc_list.size = 0;
        out.odsc_list.raw_odsc = NULL;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);

    } else {
        odsc_tab = malloc(sizeof(*odsc_tab) * num_odsc);
        for(int j = 0; j < num_odsc; j++) {
            obj_descriptor odsc;
            odsc = *podsc[j];
            DEBUG_OUT("including %s\n", obj_desc_sprint(&odsc));
            /* Preserve storage type at the destination. */
            odsc.st = in_odsc.st;
            bbox_intersect(&in_odsc.bb, &odsc.bb, &odsc.bb);
            odsc_tab[j] = odsc;
        }
        out.odsc_list.size = num_odsc * sizeof(obj_descriptor);
        out.odsc_list.raw_odsc = (char *)odsc_tab;
        margo_irespond(handle, &out, &req);
        DEBUG_OUT("sent response...waiting on request handle\n");
        margo_free_input(handle, &in);
        margo_wait(req);
        DEBUG_OUT("request handle complete.\n");
        margo_destroy(handle);
    }
    DEBUG_OUT("complete\n");

    free(podsc);
}
DEFINE_MARGO_RPC_HANDLER(odsc_internal_rpc)

/*
  Rpc routine to update (add or insert) an object descriptor in the
  dht table.
*/
static void obj_update_rpc(hg_handle_t handle)
{
    hg_return_t hret;
    odsc_gdim_t in;
    obj_update_t type;
    int err;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("Received rpc to update obj_dht\n");

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    obj_descriptor in_odsc;
    memcpy(&in_odsc, in.odsc_gdim.raw_odsc, sizeof(in_odsc));
    struct global_dimension gdim;
    memcpy(&gdim, in.odsc_gdim.raw_gdim, sizeof(struct global_dimension));
    type = in.param;

    DEBUG_OUT("received update_rpc %s\n", obj_desc_sprint(&in_odsc));
    ABT_mutex_lock(server->sspace_mutex);
    DEBUG_OUT("got sspace lock.\n");
    struct sspace *ssd = lookup_sspace(server, in_odsc.name, &gdim);
    ABT_mutex_unlock(server->sspace_mutex);
    struct dht_entry *de = ssd->ent_self;

    ABT_mutex_lock(server->dht_mutex);
    switch(type) {
    case DS_OBJ_NEW:
        err = dht_add_entry(de, &in_odsc);
        break;
    case DS_OBJ_OWNER:
        err = dht_update_owner(de, &in_odsc, 1);
        break;
    default:
        fprintf(stderr, "ERROR: (%s): unknown object update type.\n", __func__);
    }
    ABT_mutex_unlock(server->dht_mutex);
    DEBUG_OUT("Updated dht %s in server %d \n", obj_desc_sprint(&in_odsc),
              server->dsg->rank);
    if(err < 0)
        fprintf(stderr, "ERROR (%s): obj_update_rpc Failed with %d\n", __func__,
                err);

    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(obj_update_rpc)

static void ss_rpc(hg_handle_t handle)
{
    ss_information out;

    margo_instance_id mid = margo_hg_handle_get_instance(handle);

    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);

    DEBUG_OUT("received ss_rpc\n");

    ss_info_hdr ss_data;
    ss_data.num_dims = server->conf.ndim;
    ss_data.num_space_srv = server->dsg->size_sp;
    ss_data.max_versions = server->conf.max_versions;
    ss_data.hash_version = server->conf.hash_version;
    ss_data.default_gdim.ndim = server->conf.ndim;

    for(int i = 0; i < server->conf.ndim; i++) {
        ss_data.ss_domain.lb.c[i] = 0;
        ss_data.ss_domain.ub.c[i] = server->conf.dims.c[i] - 1;
        ss_data.default_gdim.sizes.c[i] = server->conf.dims.c[i];
    }

    out.ss_buf.size = sizeof(ss_info_hdr);
    out.ss_buf.raw_odsc = (char *)(&ss_data);
    out.chk_str = strdup("chkstr");
    margo_respond(handle, &out);
    DEBUG_OUT("responded in %s\n", __func__);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(ss_rpc)

static void send_kill_rpc(dspaces_provider_t server, int target, int *rank)
{
    // TODO: error handling/reporting
    hg_addr_t server_addr;
    hg_handle_t h;
    margo_request req;

    margo_addr_lookup(server->mid, server->server_address[target],
                      &server_addr);
    margo_create(server->mid, server_addr, server->kill_id, &h);
    margo_iforward(h, rank, &req);
    margo_addr_free(server->mid, server_addr);
    margo_destroy(h);
}

static void kill_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    struct ibcast_state *bcast;
    int32_t src, rank;
    int do_kill = 0;

    margo_get_input(handle, &src);
    DEBUG_OUT("Received kill signal from %d.\n", src);

    rank = server->dsg->rank;

    ABT_mutex_lock(server->kill_mutex);
    DEBUG_OUT("Kill tokens remaining: %d\n",
              server->f_kill ? (server->f_kill - 1) : 0);
    if(server->f_kill == 0) {
        // already shutting down
        ABT_mutex_unlock(server->kill_mutex);
        margo_free_input(handle, &src);
        margo_destroy(handle);
        return;
    }
    if(--server->f_kill == 0) {
        DEBUG_OUT("Kill count is zero. Initiating shutdown.\n");
        do_kill = 1;
    }

    ABT_mutex_unlock(server->kill_mutex);

    bcast = ibcast_rpc_start(server, server->kill_id, src, &rank);
    if(bcast) {
        ibcast_finish(server, bcast);
    }

    margo_free_input(handle, &src);
    margo_destroy(handle);
    if(do_kill) {
        server_destroy(server);
    }
    DEBUG_OUT("finished with kill handling.\n");
}
DEFINE_MARGO_RPC_HANDLER(kill_rpc)

static void sub_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    odsc_list_t notice;
    odsc_gdim_t in;
    int32_t sub_id;
    obj_descriptor in_odsc;
    obj_descriptor *results;
    struct global_dimension in_gdim;
    hg_addr_t client_addr;
    hg_handle_t notifyh;
    margo_request req;
    static int uid = 0;
    int req_id;

    req_id = __sync_fetch_and_add(&uid, 1L);
    margo_get_input(handle, &in);

    memcpy(&in_odsc, in.odsc_gdim.raw_odsc, sizeof(in_odsc));
    memcpy(&in_gdim, in.odsc_gdim.raw_gdim, sizeof(struct global_dimension));
    sub_id = in.param;

    DEBUG_OUT("received subscription for %s with id %d from %s\n",
              obj_desc_sprint(&in_odsc), sub_id, in_odsc.owner);

    in.param = -1; // this will be interpreted as timeout by any interal queries
    notice.odsc_list.size = get_query_odscs(server, &in, -1, &results, req_id);
    notice.odsc_list.raw_odsc = (char *)results;
    notice.param = sub_id;

    margo_addr_lookup(server->mid, in_odsc.owner, &client_addr);
    margo_create(server->mid, client_addr, server->notify_id, &notifyh);
    margo_iforward(notifyh, &notice, &req);
    DEBUG_OUT("send reply for req_id %i\n", req_id);
    margo_addr_free(server->mid, client_addr);
    margo_destroy(notifyh);

    margo_free_input(handle, &in);
    margo_destroy(handle);

    free(results);
}
DEFINE_MARGO_RPC_HANDLER(sub_rpc)

static void do_ops_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    do_ops_in_t in;
    bulk_out_t out;
    struct obj_data *od, *stage_od, *res_od;
    struct list_head odl;
    struct ds_data_expr *expr;
    struct global_dimension *gdim;
    obj_descriptor *odsc;
    int res_size;
    int num_odscs;
    obj_descriptor *q_results;
    uint64_t res_buf_size;
    odsc_gdim_t query;
    hg_return_t hret;
    void *buffer, *cbuffer;
    hg_bulk_t bulk_handle;
    hg_size_t size;
    int err;
    int csize;
    long i;

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: %s: margo_get_input() failed with %d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }
    expr = in.expr;
    DEBUG_OUT("doing expression type %i\n", expr->type);

    INIT_LIST_HEAD(&odl);
    gather_op_ods(expr, &odl);
    list_for_each_entry(od, &odl, struct obj_data, obj_entry)
    {
        odsc = &od->obj_desc;
        DEBUG_OUT("Finding data for '%s'\n", odsc->name);
        res_od = obj_data_alloc(odsc);
        stage_od = ls_find(server->dsg->ls, odsc);
        if(!stage_od) {
            DEBUG_OUT("not stored locally.\n");
            // size is currently not used`
            query.odsc_gdim.size = sizeof(*odsc);
            query.odsc_gdim.raw_odsc = (char *)odsc;
            query.odsc_gdim.gdim_size = sizeof(od->gdim);
            query.odsc_gdim.raw_gdim = (char *)&od->gdim;

            // TODO: assumes data is either local or on a remote
            res_size = get_query_odscs(server, &query, -1, &q_results, -1);
            if(res_size != sizeof(odsc)) {
                fprintf(stderr, "WARNING: %s: multiple odscs for query.\n",
                        __func__);
            }
            stage_od = ls_find(server->dsg->ps, odsc);
            if(!stage_od) {
                fprintf(stderr,
                        "ERROR: %s: nothing in the proxy cache for query.\n",
                        __func__);
            }
        }
        ssd_copy(res_od, stage_od);
        // update any obj in expression to use res_od
        DEBUG_OUT("updating expression data with variable data.\n");
        update_expr_objs(expr, res_od);
    }

    res_buf_size = expr->size;
    buffer = malloc(res_buf_size);
    cbuffer = malloc(res_buf_size);
    if(expr->type == DS_VAL_INT) {
        DEBUG_OUT("Executing integer operation on %" PRIu64 " elements\n",
                  res_buf_size / sizeof(int));
#pragma omp for
        for(i = 0; i < res_buf_size / sizeof(int); i++) {
            ((int *)buffer)[i] = ds_op_calc_ival(expr, i, &err);
        }
    } else if(expr->type == DS_VAL_REAL) {
        DEBUG_OUT("Executing real operation on %" PRIu64 " elements\n",
                  res_buf_size / sizeof(double));
#pragma omp for
        for(i = 0; i < res_buf_size / sizeof(double); i++) {
            ((double *)buffer)[i] = ds_op_calc_rval(expr, i, &err);
        }
    } else {
        fprintf(stderr, "ERROR: %s: invalid expression data type.\n", __func__);
        goto cleanup;
    }
    size = res_buf_size;
    hret = margo_bulk_create(mid, 1, (void **)&cbuffer, &size,
                             HG_BULK_READ_ONLY, &bulk_handle);

    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_create() failure\n", __func__);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
    }

    csize = LZ4_compress_default(buffer, cbuffer, size, size);

    DEBUG_OUT("compressed result from %" PRIu64 " to %i bytes.\n", size, csize);
    if(!csize) {
        DEBUG_OUT("compressed result could not fit in dst buffer - longer than "
                  "original! Sending uncompressed.\n");
        memcpy(cbuffer, buffer, size);
    }

    hret = margo_bulk_transfer(mid, HG_BULK_PUSH, info->addr, in.handle, 0,
                               bulk_handle, 0, (csize ? csize : size));
    if(hret != HG_SUCCESS) {
        fprintf(stderr, "ERROR: (%s): margo_bulk_transfer() failure (%d)\n",
                __func__, hret);
        out.ret = dspaces_ERR_MERCURY;
        margo_respond(handle, &out);
        margo_bulk_free(bulk_handle);
        goto cleanup;
    }
    margo_bulk_free(bulk_handle);
    out.ret = dspaces_SUCCESS;
    out.len = csize;
    margo_respond(handle, &out);
cleanup:
    free(buffer);
    free(cbuffer);
    margo_free_input(handle, &in);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(do_ops_rpc)

#ifdef DSPACES_HAVE_PYTHON
static PyObject *build_ndarray_from_od(struct obj_data *od)
{
    obj_descriptor *odsc = odsc = &od->obj_desc;
    int tag = odsc->tag;
    PyArray_Descr *descr = PyArray_DescrNewFromType(tag);
    int ndim = odsc->bb.num_dims;
    void *data = od->data;
    PyObject *arr, *fnp;
    npy_intp dims[ndim];
    int i;

    for(i = 0; i < ndim; i++) {
        dims[(ndim - i) - 1] = (odsc->bb.ub.c[i] - odsc->bb.lb.c[i]) + 1;
    }

    arr = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, dims, NULL, data, 0,
                               NULL);

    return (arr);
}

static void pexec_rpc(hg_handle_t handle)
{
    PyGILState_STATE gstate;
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    pexec_in_t in;
    pexec_out_t out;
    hg_return_t hret;
    hg_bulk_t bulk_handle;
    obj_descriptor in_odsc, odsc;
    hg_size_t rdma_size;
    void *fn = NULL, *res_data;
    struct obj_data *od, *arg_obj, **od_tab, **from_objs = NULL;
    int num_obj;
    PyObject *array, *fnp, *arg, *pres, *pres_bytes;
    static PyObject *pklmod = NULL;
    ABT_cond cond;
    ABT_mutex mtx;
    int i;

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    memcpy(&in_odsc, in.odsc.raw_odsc, sizeof(in_odsc));

    DEBUG_OUT("received pexec request\n");
    rdma_size = in.length;
    if(rdma_size > 0) {
        DEBUG_OUT("function included, length %" PRIu32 " bytes\n", in.length);
        fn = malloc(rdma_size);
        hret = margo_bulk_create(mid, 1, (void **)&(fn), &rdma_size,
                                 HG_BULK_WRITE_ONLY, &bulk_handle);

        if(hret != HG_SUCCESS) {
            // TODO: communicate failure
            fprintf(stderr, "ERROR: (%s): margo_bulk_create failed!\n",
                    __func__);
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_destroy(handle);
            return;
        }

        hret = margo_bulk_transfer(mid, HG_BULK_PULL, info->addr, in.handle, 0,
                                   bulk_handle, 0, rdma_size);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_transfer failed!\n",
                    __func__);
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_bulk_free(bulk_handle);
            margo_destroy(handle);
            return;
        }
    }
    margo_bulk_free(bulk_handle);
    route_request(server, &in_odsc, &(server->dsg->default_gdim));

    ABT_mutex_lock(server->ls_mutex);
    num_obj = ls_find_all(server->dsg->ls, &in_odsc, &from_objs);
    if(num_obj > 0) {
        DEBUG_OUT("found %i objects in local storage to populate input\n",
                  num_obj);
        arg_obj = obj_data_alloc(&in_odsc);
        od_tab = malloc(num_obj * sizeof(*od_tab));
        for(i = 0; i < num_obj; i++) {
            // Can we skip the intermediate copy?
            odsc = from_objs[i]->obj_desc;
            bbox_intersect(&in_odsc.bb, &odsc.bb, &odsc.bb);
            od_tab[i] = obj_data_alloc(&odsc);
            ssd_copy(od_tab[i], from_objs[i]);
            ssd_copy(arg_obj, od_tab[i]);
            obj_data_free(od_tab[i]);
        }
        free(od_tab);
    }
    ABT_mutex_unlock(server->ls_mutex);
    if(num_obj < 1) {
        DEBUG_OUT("could not find input object\n");
        out.length = 0;
        out.handle = 0;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        return;
    } else if(from_objs) {
        free(from_objs);
    }
    gstate = PyGILState_Ensure();
    array = build_ndarray_from_od(arg_obj);

    if((pklmod == NULL) && (pklmod = PyImport_ImportModule("dill")) == NULL) {
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        PyGILState_Release(gstate);
        return;
    }
    arg = PyBytes_FromStringAndSize(fn, rdma_size);
    fnp = PyObject_CallMethodObjArgs(pklmod, PyUnicode_FromString("loads"), arg,
                                     NULL);
    Py_XDECREF(arg);
    if(fnp && PyCallable_Check(fnp)) {
        pres = PyObject_CallFunctionObjArgs(fnp, array, NULL);
    } else {
        if(!fnp) {
            PyErr_Print();
        }
        fprintf(stderr,
                "ERROR: (%s): provided function could either not be loaded, or "
                "is not callable.\n",
                __func__);
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        PyGILState_Release(gstate);
        return;
    }

    if(pres && (pres != Py_None)) {
        pres_bytes = PyObject_CallMethodObjArgs(
            pklmod, PyUnicode_FromString("dumps"), pres, NULL);
        Py_XDECREF(pres);
        res_data = PyBytes_AsString(pres_bytes);
        rdma_size = PyBytes_Size(pres_bytes) + 1;
        hret = margo_bulk_create(mid, 1, (void **)&res_data, &rdma_size,
                                 HG_BULK_READ_ONLY, &out.handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_create failed with %d.\n",
                    __func__, hret);
            out.length = 0;
            out.handle = 0;
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_destroy(handle);
            PyGILState_Release(gstate);
            return;
        }
        out.length = rdma_size;
    } else {
        if(!pres) {
            PyErr_Print();
        }
        out.length = 0;
    }
    PyGILState_Release(gstate);

    if(out.length > 0) {
        ABT_cond_create(&cond);
        ABT_mutex_create(&mtx);
        out.condp = (uint64_t)(&cond);
        out.mtxp = (uint64_t)(&mtx);
        DEBUG_OUT("sending out.condp = %" PRIu64 " and out.mtxp = %" PRIu64
                  "\n",
                  out.condp, out.mtxp);
        ABT_mutex_lock(mtx);
        margo_respond(handle, &out);
        ABT_cond_wait(cond, mtx);
        DEBUG_OUT("signaled on condition\n");
        ABT_mutex_unlock(mtx);
        ABT_mutex_free(&mtx);
        ABT_cond_free(&cond);
    } else {
        out.handle = 0;
        margo_respond(handle, &out);
    }

    margo_free_input(handle, &in);
    margo_destroy(handle);

    DEBUG_OUT("done with pexec handling\n");

    gstate = PyGILState_Ensure();
    Py_XDECREF(array);
    PyGILState_Release(gstate);
    obj_data_free(arg_obj);
}
DEFINE_MARGO_RPC_HANDLER(pexec_rpc)

static void mpexec_rpc(hg_handle_t handle)
{
    PyGILState_STATE gstate;
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    pexec_in_t in;
    pexec_out_t out;
    hg_return_t hret;
    hg_bulk_t bulk_handle;
    obj_descriptor *in_odsc, *in_odscs, odsc;
    int num_args;
    hg_size_t rdma_size;
    void *fn = NULL, *res_data;
    struct obj_data *od, **arg_objs, *arg_obj, **od_tab, **from_objs = NULL;
    int num_obj;
    PyObject **arg_arrays, *fnp, *arg, *args, *pres, *pres_bytes;
    static PyObject *pklmod = NULL;
    ABT_cond cond;
    ABT_mutex mtx;
    int i, j;

    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }
    in_odscs = (obj_descriptor *)in.odsc.raw_odsc;
    num_args = in.odsc.size / sizeof(*in_odscs);

    DEBUG_OUT("received mpexec request with %i args\n", num_args);
    rdma_size = in.length;
    if(rdma_size > 0) {
        DEBUG_OUT("function included, length %" PRIu32 " bytes\n", in.length);
        fn = malloc(rdma_size);
        hret = margo_bulk_create(mid, 1, (void **)&(fn), &rdma_size,
                                 HG_BULK_WRITE_ONLY, &bulk_handle);

        if(hret != HG_SUCCESS) {
            // TODO: communicate failure
            fprintf(stderr, "ERROR: (%s): margo_bulk_create failed!\n",
                    __func__);
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_destroy(handle);
            return;
        }

        hret = margo_bulk_transfer(mid, HG_BULK_PULL, info->addr, in.handle, 0,
                                   bulk_handle, 0, rdma_size);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_transfer failed!\n",
                    __func__);
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_bulk_free(bulk_handle);
            margo_destroy(handle);
            return;
        }
    }
    margo_bulk_free(bulk_handle);

    arg_objs = calloc(num_args, sizeof(*arg_objs));
    arg_arrays = calloc(num_args, sizeof(*arg_arrays));
    for(i = 0; i < num_args; i++) {
        route_request(server, &in_odscs[i], &(server->dsg->default_gdim));

        in_odsc = &in_odscs[i];
        DEBUG_OUT("searching for local storage objects to satisfy %s\n",
                  obj_desc_sprint(in_odsc));
        ABT_mutex_lock(server->ls_mutex);
        num_obj = ls_find_all(server->dsg->ls, in_odsc, &from_objs);
        if(num_obj > 0) {
            // need one source of information - metadata case?
            if(in_odsc->size == 0) {
                in_odsc->size = from_objs[0]->obj_desc.size;
            }
            DEBUG_OUT("found %i objects in local storage to populate input\n",
                      num_obj);
            arg_objs[i] = obj_data_alloc(in_odsc);
            od_tab = malloc(num_obj * sizeof(*od_tab));
            for(j = 0; j < num_obj; j++) {
                // Can we skip the intermediate copy?
                odsc = from_objs[j]->obj_desc;
                DEBUG_OUT("getting data from %s\n", obj_desc_sprint(&odsc));
                bbox_intersect(&in_odsc->bb, &odsc.bb, &odsc.bb);
                od_tab[j] = obj_data_alloc(&odsc);
                DEBUG_OUT("overlap = %s\n",
                          obj_desc_sprint(&od_tab[j]->obj_desc));
                ssd_copy(od_tab[j], from_objs[j]);
                ssd_copy(arg_objs[i], od_tab[j]);
                obj_data_free(od_tab[j]);
            }
            free(od_tab);
        }
        ABT_mutex_unlock(server->ls_mutex);
        if(num_obj < 1) {
            DEBUG_OUT("could not find input object\n");
            out.length = 0;
            out.handle = 0;
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_destroy(handle);
            free(arg_arrays);
            for(j = 0; j < i; j++) {
                obj_data_free(arg_objs[j]);
            }
            free(arg_objs);
            return;
        } else if(from_objs) {
            free(from_objs);
        }

        arg_arrays[i] = build_ndarray_from_od(arg_objs[i]);
        DEBUG_OUT("created ndarray for %s\n",
                  obj_desc_sprint(&arg_objs[i]->obj_desc));
    }

    gstate = PyGILState_Ensure();
    if((pklmod == NULL) &&
       (pklmod = PyImport_ImportModuleNoBlock("dill")) == NULL) {
        out.length = 0;
        out.handle = 0;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        free(arg_arrays);
        free(arg_objs);
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        PyGILState_Release(gstate);
        return;
    }
    arg = PyBytes_FromStringAndSize(fn, rdma_size);
    fnp = PyObject_CallMethodObjArgs(pklmod, PyUnicode_FromString("loads"), arg,
                                     NULL);
    Py_XDECREF(arg);

    args = PyTuple_New(num_args);
    for(i = 0; i < num_args; i++) {
        PyTuple_SetItem(args, i, arg_arrays[i]);
    }
    if(fnp && PyCallable_Check(fnp)) {
        pres = PyObject_CallObject(fnp, args);
    } else {
        if(!fnp) {
            PyErr_Print();
        }
        fprintf(stderr,
                "ERROR: (%s): provided function could either not be loaded, or "
                "is not callable.\n",
                __func__);
        out.length = 0;
        out.handle = 0;
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        free(arg_arrays);
        free(arg_objs);
        margo_respond(handle, &out);
        margo_free_input(handle, &in);
        margo_destroy(handle);
        PyGILState_Release(gstate);
        return;
    }

    if(pres && (pres != Py_None)) {
        pres_bytes = PyObject_CallMethodObjArgs(
            pklmod, PyUnicode_FromString("dumps"), pres, NULL);
        Py_XDECREF(pres);
        res_data = PyBytes_AsString(pres_bytes);
        rdma_size = PyBytes_Size(pres_bytes) + 1;
        hret = margo_bulk_create(mid, 1, (void **)&res_data, &rdma_size,
                                 HG_BULK_READ_ONLY, &out.handle);
        if(hret != HG_SUCCESS) {
            fprintf(stderr, "ERROR: (%s): margo_bulk_create failed with %d.\n",
                    __func__, hret);
            out.length = 0;
            out.handle = 0;
            margo_respond(handle, &out);
            margo_free_input(handle, &in);
            margo_destroy(handle);
            PyGILState_Release(gstate);
            return;
        }
        out.length = rdma_size;
    } else {
        if(!pres) {
            PyErr_Print();
        }
        out.length = 0;
    }
    PyGILState_Release(gstate);

    if(out.length > 0) {
        ABT_cond_create(&cond);
        ABT_mutex_create(&mtx);
        out.condp = (uint64_t)(&cond);
        out.mtxp = (uint64_t)(&mtx);
        DEBUG_OUT("sending out.condp = %" PRIu64 " and out.mtxp = %" PRIu64
                  "\n",
                  out.condp, out.mtxp);
        ABT_mutex_lock(mtx);
        margo_respond(handle, &out);
        ABT_cond_wait(cond, mtx);
        DEBUG_OUT("signaled on condition\n");
        ABT_mutex_unlock(mtx);
        ABT_mutex_free(&mtx);
        ABT_cond_free(&cond);
    } else {
        out.handle = 0;
        margo_respond(handle, &out);
    }

    margo_free_input(handle, &in);
    margo_destroy(handle);

    DEBUG_OUT("done with pexec handling\n");

    gstate = PyGILState_Ensure();
    Py_XDECREF(args);
    PyGILState_Release(gstate);
    free(arg_arrays);
    if(num_args) {
        for(i = 0; i < num_args; i++) {
            obj_data_free(arg_objs[i]);
        }
        free(arg_objs);
    }
}
DEFINE_MARGO_RPC_HANDLER(mpexec_rpc)

#endif // DSPACES_HAVE_PYTHON

static void cond_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    ABT_mutex *mtx;
    ABT_cond *cond;
    cond_in_t in;

    margo_get_input(handle, &in);
    DEBUG_OUT("condition rpc for mtxp = %" PRIu64 ", condp = %" PRIu64 "\n",
              in.mtxp, in.condp);

    mtx = (ABT_mutex *)in.mtxp;
    cond = (ABT_cond *)in.condp;
    ABT_mutex_lock(*mtx);
    ABT_cond_signal(*cond);
    ABT_mutex_unlock(*mtx);

    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(cond_rpc)

static void get_vars_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    struct ibcast_state *bcast;
    int32_t src, rank;
    name_list_t out[3] = {0};
    name_list_t rout = {0};
    ds_str_hash *results = ds_str_hash_init();
    int present;
    int num_vars;
    char **names = NULL;
    int i, j;

    margo_get_input(handle, &src);
    DEBUG_OUT("Received request for all variable names from %d\n", src);

    rank = server->dsg->rank;
    bcast = ibcast_rpc_start(server, server->get_vars_id, src, &rank);

    ABT_mutex_lock(server->ls_mutex);
    num_vars = ls_get_var_names(server->dsg->ls, &names);
    ABT_mutex_unlock(server->ls_mutex);

    DEBUG_OUT("found %d local variables.\n", num_vars);

    for(i = 0; i < num_vars; i++) {
        ds_str_hash_add(results, names[i]);
        free(names[i]);
    }
    if(names) {
        free(names);
    }

    if(bcast) {
        for(i = 0; i < 3; i++) {
            ibcast_get_output(bcast, i, &out[i], &present);
            if(present) {
                for(j = 0; j < out[i].count; j++) {
                    ds_str_hash_add(results, out[i].names[j]);
                }
            }
        }
        ibcast_finish(server, bcast);
    }

    rout.count = ds_str_hash_get_all(results, &rout.names);
    DEBUG_OUT("returning %" PRIu64 " variable names\n", rout.count);
    margo_respond(handle, &rout);
    for(i = 0; i < rout.count; i++) {
        free(rout.names[i]);
    }
    if(rout.count > 0) {
        free(rout.names);
    }
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(get_vars_rpc)

static void get_var_objs_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    get_var_objs_in_t in, bcast_in;
    struct ibcast_state *bcast;
    int32_t src, rank;
    odsc_hdr out[3];
    obj_descriptor **odsc_tab;
    int i;
    int num_odscs;
    int present;
    odsc_hdr rout;
    hg_return_t hret;

    margo_get_input(handle, &in);
    src = in.src;

    DEBUG_OUT("Received request for all objects of '%s' from %d\n", in.var_name,
              src);

    rank = server->dsg->rank;

    bcast_in.src = rank;
    bcast_in.var_name = in.var_name;

    bcast = ibcast_rpc_start(server, server->get_var_objs_id, src, &bcast_in);

    ABT_mutex_lock(server->ls_mutex);
    num_odscs = ls_find_all_no_version(server->dsg->ls, in.var_name, &odsc_tab);
    rout.size = num_odscs * sizeof(obj_descriptor);
    if(rout.size > 0) {
        rout.raw_odsc = malloc(rout.size);
        for(i = 0; i < num_odscs; i++) {
            ((obj_descriptor *)rout.raw_odsc)[i] = *(odsc_tab[i]);
        }
        free(odsc_tab);
    } else {
        rout.raw_odsc = NULL;
    }
    ABT_mutex_unlock(server->ls_mutex);

    DEBUG_OUT("found %d object descriptors locally.\n", num_odscs);

    if(bcast) {
        for(i = 0; i < 3; i++) {
            ibcast_get_output(bcast, i, &out[i], &present);
            if(present) {
                rout.raw_odsc = realloc(rout.raw_odsc, rout.size + out[i].size);
                memcpy(rout.raw_odsc + rout.size, out[i].raw_odsc, out[i].size);
                rout.size += out[i].size;
            }
        }
        ibcast_finish(server, bcast);
    }

    DEBUG_OUT("returning %zi objects.\n", rout.size / sizeof(obj_descriptor));
    margo_respond(handle, &rout);

    if(rout.size) {
        free(rout.raw_odsc);
    }

    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(get_var_objs_rpc)

static int dspaces_init_registry(dspaces_provider_t server)
{
    struct dspaces_module *mod;
    struct dspaces_module_args arg;
    struct dspaces_module_ret *res = NULL;
    int err;

    server->local_reg_id = 0;

    mod = dspaces_mod_by_name(&server->mods, "ds_reg");

    if(!mod) {
        fprintf(stderr, "missing built-in ds_reg module.\n");
        return (-1);
    }

    build_module_arg_from_rank(server->rank, &arg);
    res =
        dspaces_module_exec(mod, "bootstrap_id", &arg, 1, DSPACES_MOD_RET_INT);
    if(res && res->type == DSPACES_MOD_RET_ERR) {
        err = res->err;
        free(res);
        return (err);
    }
    if(res) {
        server->local_reg_id = res->ival;
    }

    DEBUG_OUT("local registry id starting at %i\n", server->local_reg_id);

    return (0);
}

static int route_registration(dspaces_provider_t server, reg_in_t *reg)
{
    struct dspaces_module *mod, *reg_mod;
    int nargs;
    struct dspaces_module_ret *res = NULL;
    struct dspaces_module_args *args;
    int err;

    DEBUG_OUT("routing registration request.\n");

    mod = dspaces_mod_by_name(&server->mods, reg->type);
    if(!mod) {
        DEBUG_OUT("could not find module for type '%s'.\n", reg->type);
        return (DS_MOD_ENOMOD);
    }

    // ds_reg module doesn't have access to the map between dspaces module name
    // and Python module name, so translate for it.
    free(reg->type);
    reg->type = strdup(mod->file);

    reg_mod = dspaces_mod_by_name(&server->mods, "ds_reg");
    if(!reg_mod) {
        return (DS_MOD_EFAULT);
    }
    nargs = build_module_args_from_reg(reg, &args);
    res = dspaces_module_exec(reg_mod, "register", args, nargs,
                              DSPACES_MOD_RET_INT);
    if(!res) {
        return (DS_MOD_EFAULT);
    }
    if(res->type == DSPACES_MOD_RET_ERR) {
        err = res->err;
        free(res);
        return (err);
    }
    if(res->ival != reg->id) {
        DEBUG_OUT("updating registration id to %li.\n", res->ival);
        reg->id = res->ival;
    }

    return (0);
}

static void reg_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    hg_return_t hret;
    reg_in_t in;
    uint64_t out;
    struct ibcast_state *bcast;
    int i, err;

    mid = margo_hg_handle_get_instance(handle);
    info = margo_get_info(handle);
    server = (dspaces_provider_t)margo_registered_data(mid, info->id);
    hret = margo_get_input(handle, &in);
    if(hret != HG_SUCCESS) {
        fprintf(stderr,
                "DATASPACES: ERROR handling %s: margo_get_input() failed with "
                "%d.\n",
                __func__, hret);
        margo_destroy(handle);
        return;
    }

    DEBUG_OUT("received registration '%s' of type '%s' with %zi bytes of "
              "registration data.\n",
              in.name, in.type, strlen(in.reg_data));

    if(in.src == -1) {
        in.id = __sync_fetch_and_add(&server->local_reg_id, 1);
        in.id += (uint64_t)server->dsg->rank << 40;
    }
    bcast = ibcast_rpc_start(server, server->reg_id, in.src, &in);
    if(bcast) {
        for(i = 0; i < 3; i++) {
            ibcast_get_output(bcast, i, &out, NULL);
        }
        ibcast_finish(server, bcast);
    }

    err = route_registration(server, &in);
    if(err != 0) {
        out = err;
    } else {
        out = in.id;
    }

    margo_free_input(handle, &in);
    margo_respond(handle, &out);
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(reg_rpc);

static void get_mods_rpc(hg_handle_t handle)
{
    margo_instance_id mid = margo_hg_handle_get_instance(handle);
    const struct hg_info *info = margo_get_info(handle);
    dspaces_provider_t server =
        (dspaces_provider_t)margo_registered_data(mid, info->id);
    name_list_t out = {0};
    int i;

    DEBUG_OUT("Received request for all module names.\n");

    out.count = dspaces_module_names(&server->mods, &out.names);
    DEBUG_OUT("returning %" PRIu64 " module names\n", out.count);
    margo_respond(handle, &out);

    for(i = 0; i < out.count; i++) {
        free(out.names[i]);
    }
    if(out.count > 0) {
        free(out.names);
    }
    margo_destroy(handle);
}
DEFINE_MARGO_RPC_HANDLER(get_mods_rpc);

void dspaces_server_fini(dspaces_provider_t server)
{
    int err = 0;

    DEBUG_OUT("waiting for finalize to occur\n");
    margo_wait_for_finalize(server->mid);
#ifdef DSPACES_HAVE_PYTHON
    PyEval_RestoreThread(server->main_state);
    err = Py_FinalizeEx();
#endif // DSPACES_HAVE_PYTHON
    if(err < 0) {
        fprintf(stderr, "ERROR: Python finalize failed with %d\n", err);
    }
    DEBUG_OUT("finalize complete\n");
    free(server);
}

int dspaces_server_find_objs(dspaces_provider_t server, const char *var_name,
                             int version, struct dspaces_data_obj **objs)
{
    obj_descriptor odsc;
    obj_descriptor **od_tab;
    struct dspaces_data_obj *obj;
    int num_obj = 0;
    int i;
    ;

    strcpy(odsc.name, var_name);
    odsc.version = version;
    ABT_mutex_lock(server->ls_mutex);
    num_obj = ls_find_ods(server->dsg->ls, &odsc, &od_tab);
    ABT_mutex_unlock(server->ls_mutex);

    if(num_obj) {
        *objs = malloc(sizeof(**objs) * num_obj);
        for(i = 0; i < num_obj; i++) {
            obj = &(*objs)[i];
            obj->var_name = var_name;
            obj->version = version;
            obj->ndim = od_tab[i]->bb.num_dims;
            obj->size = od_tab[i]->size;
            obj->lb = malloc(sizeof(*obj->lb) * obj->ndim);
            obj->ub = malloc(sizeof(*obj->ub) * obj->ndim);
            memcpy(obj->lb, od_tab[i]->bb.lb.c,
                   sizeof(*obj->lb) * od_tab[i]->bb.num_dims);
            memcpy(obj->ub, od_tab[i]->bb.ub.c,
                   sizeof(*obj->ub) * od_tab[i]->bb.num_dims);
        }
        free(od_tab);
    }
    return (num_obj);
}

int dspaces_server_get_objdata(dspaces_provider_t server,
                               struct dspaces_data_obj *obj, void *buffer)
{
    obj_descriptor odsc;
    struct obj_data *od;
    int i;

    strcpy(odsc.name, obj->var_name);
    odsc.version = obj->version;
    odsc.bb.num_dims = obj->ndim;
    memcpy(odsc.bb.lb.c, obj->lb, sizeof(*obj->lb) * obj->ndim);
    memcpy(odsc.bb.ub.c, obj->ub, sizeof(*obj->ub) * obj->ndim);

    ABT_mutex_lock(server->ls_mutex);
    od = ls_find(server->dsg->ls, &odsc);
    if(!od) {
        fprintf(stderr, "WARNING: (%s): obj not found in local storage.\n",
                __func__);
        return (-1);
    } else {
        for(i = 0; i < obj->ndim; i++) {
            if(od->obj_desc.bb.lb.c[i] != obj->lb[i] ||
               od->obj_desc.bb.ub.c[i] != obj->ub[i]) {
                fprintf(stderr,
                        "WARNING: (%s): obj found, but not the right size.\n",
                        __func__);
                return (-2);
            }
        }
    }

    memcpy(buffer, od->data, obj->size * bbox_volume(&odsc.bb));
    ABT_mutex_unlock(server->ls_mutex);

    return (0);
}
