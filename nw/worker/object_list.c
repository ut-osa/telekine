#include <stdlib.h>
#include <stdio.h>

#include "common/devconf.h"
#include "object_list.h"


GHashTable *
InitObjectTable()
{
    GHashTable *objectTable;

    objectTable = g_hash_table_new(g_direct_hash, g_direct_equal);

    return objectTable;
}


VOID
InsertNewObject(GHashTable *Table, HANDLE Handle, size_t Size)
{
    PDEVICE_OBJECT_LIST newObject;

    if (GetObjectByOriginalHandle(Table, Handle) != NULL)
    {
        fprintf(stderr, "found duplicated handle %lx\n", Handle);
        return;
    }

    newObject = (PDEVICE_OBJECT_LIST)malloc(sizeof(DEVICE_OBJECT_LIST));
    newObject->OriginalObjectHandle = Handle;
    newObject->ObjectHandle = Handle;
    newObject->ObjectSize = Size;

    g_hash_table_insert(Table, GINT_TO_POINTER(Handle),
                        GINT_TO_POINTER(newObject));
}


PDEVICE_OBJECT_LIST
GetObjectByOriginalHandle(GHashTable *Table, HANDLE OriginalHandle)
{
    PDEVICE_OBJECT_LIST object;

    object = (PDEVICE_OBJECT_LIST)
        g_hash_table_lookup(Table, GINT_TO_POINTER(OriginalHandle));

    return object;
}


VOID
RemoveObject(GHashTable *Table, HANDLE Handle)
{
    gboolean found;

    found = g_hash_table_remove(Table, GINT_TO_POINTER(Handle));

    if (!found)
    {
        fprintf(stderr, "failed to find entry for handle %lx\n", Handle);
    }
}

