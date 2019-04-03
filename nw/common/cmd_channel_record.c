#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/cmd_channel.h"

enum command_channel_type { AVA_LOCAL, AVA_SHM, AVA_VSOCK };

struct command_channel_record {
    enum command_channel_type chan_type;
    FILE *frecord; // TODO: use file descriptor to create unique interface for network streaming.
    FILE *fload;
};

struct record_command_metadata {
    size_t size;
    uint32_t flags; // TODO: we can set flags such as invalid bit after the command is recorded.
};

//!-- Record APIs

/**
 * Serialize the command to a file descriptor and convert shared memory
 * commnd format into socket command format.
 *
 */
ssize_t command_channel_record_command(struct command_channel* c, struct command_base* cmd)
{
    struct command_channel_record *chan = (struct command_channel_record *)c;
    ssize_t pos = ftell(chan->frecord);
    void *cmd_data_region = command_channel_get_data_region(c, cmd);
    struct record_command_metadata metadata = {
        .size = cmd->command_size + cmd->region_size,
        .flags = 0,
    };

    fwrite(&metadata, sizeof(struct record_command_metadata), 1, chan->frecord);
    fwrite(cmd, cmd->command_size, 1, chan->frecord);
    fwrite(cmd_data_region, cmd->region_size, 1, chan->frecord);

    return pos;
}

/**
 * Update the flags of a recorded command. The offset in the record file
 * must be non-negative. This function is exposed only if the file
 * descriptor is seekable.
 */
void command_channel_record_update_flags(struct command_channel* c, ssize_t offset, uint32_t flags)
{
    struct command_channel_record *chan = (struct command_channel_record *)c;

    ssize_t pos = ftell(chan->frecord);
    assert(offset >= 0);
    fseek(chan->fload, offset + sizeof(size_t), SEEK_SET);
    fwrite(&flags, sizeof(uint32_t), 1, chan->frecord);
    fseek(chan->frecord, 0, SEEK_END);
}

struct command_channel *command_channel_record_new(int worker_port)
{
    struct command_channel_record *chan = (struct command_channel_record *)malloc(sizeof(struct command_channel_record));
    char fname[64];

    time_t now;
    time(&now);
    struct tm *local_time = localtime(&now);

    /* open log file */
    //sprintf(fname, "record_log_worker_%d_%02d-%02d-%02d.bin", worker_port,
    //        local_time->tm_hour, local_time->tm_min, local_time->tm_sec);
    sprintf(fname, "record_log_worker_%d.bin", worker_port);
    chan->frecord = fopen(fname, "wb+");
    DEBUG_PRINT("file %s created for recording\n", fname);

    /* check channel type */
    const char *ava_env = getenv("AVA_CHANNEL");
    if (!ava_env || !strcmp(ava_env, "LOCAL"))
        chan->chan_type = AVA_LOCAL;
    else if (!strcmp(ava_env, "SHM"))
        chan->chan_type = AVA_SHM;
    else if (!strcmp(ava_env, "VSOCK"))
        chan->chan_type = AVA_VSOCK;
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK]\n");
        return NULL;
    }

    return (struct command_channel *)chan;
}

void command_channel_record_free(struct command_channel* c)
{
    struct command_channel_record *chan = (struct command_channel_record *)c;
    fclose(chan->frecord);
}

//!-- Load APIs

/**
 * Load command from the offset from the beginning of the file. If the
 * offset value is negative, it continues from the last position. The
 * offset can be non-negative only if the file descriptor is seekable.
 */
struct command_base *command_channel_load_command(struct command_channel* c, ssize_t offset)
{
    struct command_channel_record *chan = (struct command_channel_record *)c;
    struct record_command_metadata metadata;
    struct command_base *cmd;

    if (offset > 0)
        fseek(chan->fload, offset, SEEK_SET);

    fread(&metadata, sizeof(struct record_command_metadata), 1, chan->fload);
    cmd = (struct command_base *)malloc(metadata.size);
    fread(cmd, metadata.size, 1, chan->fload);

    return cmd;
}

/**
 * Free the loaded command.
 */
void *command_channel_load_free_command(struct command_channel* c, struct command_base *cmd)
{
    free(cmd);
}

/**
 * Translate a buffer_id in the recorded command into a data pointer.
 * The returned pointer will be valid until `command_channel_load_free_command`
 * is called on `cmd`.
 */
void* command_channel_load_get_buffer(struct command_channel* chan, struct command_base* cmd, void* buffer_id) {
    return (void *)((uintptr_t)cmd + buffer_id);
}


struct command_channel *command_channel_load_new(int worker_port)
{
    struct command_channel_record *chan = (struct command_channel_record *)malloc(sizeof(struct command_channel_record));
    char fname[64];

    /* open log file */
    // TODO: the new worker should receive the record file name from
    // hypervisor or original worker.
    sprintf(fname, "record_log_worker_%d.bin", worker_port);
    chan->fload = fopen(fname, "rb");
    DEBUG_PRINT("file %s opened for loading\n", fname);

    /* check channel type */
    const char *ava_env = getenv("AVA_CHANNEL");
    if (!ava_env || !strcmp(ava_env, "LOCAL"))
        chan->chan_type = AVA_LOCAL;
    else if (!strcmp(ava_env, "SHM"))
        chan->chan_type = AVA_SHM;
    else if (!strcmp(ava_env, "VSOCK"))
        chan->chan_type = AVA_VSOCK;
    else {
        printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK]\n");
        return NULL;
    }

    return (struct command_channel *)chan;
}

void command_channel_load_free(struct command_channel* c)
{
    struct command_channel_record *chan = (struct command_channel_record *)c;
    fclose(chan->fload);
}
