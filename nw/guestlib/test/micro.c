#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>

//#include "common/devconf.h"
//#include "common/ioctl.h"
//#include "common/kvm.h"

int main()
{
    uintptr_t *x = (uintptr_t *)malloc(sizeof(uintptr_t));
    printf("size=%lx\n", sizeof(*x));
    uint8_t *y = (uint8_t *)x;
    printf("size=%lx\n", sizeof(*y));
    return 0;

    /*
    char dev_filename[32];
    sprintf(dev_filename, "/dev/%s%d", VGPU_DEV_NAME, VGPU_DRIVER_MINOR);

    int fd;
    fd = open(dev_filename, O_RDWR);
    if (fd < 0) {
        printf("failed to open device %s\n", dev_filename);
        exit(-1);
    }
    */

    /* setup driver mode */
    /*
    ArgList *arglist = malloc(sizeof(ArgList) + 2 * sizeof(ArgAttr));
    arglist->size = sizeof(ArgList) + 2 * sizeof(ArgAttr);
    arglist->argc = 2;
    printf("argsize = %ld\n", arglist->size);

    int ret_val;
    ret_val = ioctl(fd, IOCTL_SEND_VMCALL, (uintptr_t)arglist);

    return 0;
    */
}
