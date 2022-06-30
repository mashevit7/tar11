#include "kernel/types.h"
#include "user/user.h"
#include "kernel/fcntl.h"

int copy(int fd_src, int fd_dst) {
    unsigned char buffer[1024];
    int r = 0;
    int e = 0;
    while (0 < (r = read(fd_src, buffer, 1024)) && r <= 1024) {
        e = write(fd_dst, buffer, r);
        if (e < 0) {
            printf("error writing to the destination file.");
            return -3;
        }
    }
    if (read < 0) {
        printf("error reading from the source file.");
        return -4;
    }
    return 0;
}

int
_main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("usage: cp <src> <dest>\n");
        return 0;
    }

    int fd_src = open(argv[1], O_RDONLY);
    if (fd_src < 0) {
        printf("file '%s' does not exist or is a directory.\n", argv[1]);
        return -1;
    }
    int fd_dst = open(argv[2], O_WRONLY | O_CREATE | O_TRUNC);
    if (fd_dst < 0) {
        printf("could not open file '%s' for writing.\n", argv[2]);
        return -2;
    }
    
    return copy(fd_src, fd_dst);
}

void
main(int argc, char *argv[]) {
    exit(_main(argc, argv));
}
