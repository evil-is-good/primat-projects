#include <unistd.h>
#include <fcntl.h>
#include <linux/kd.h>
#include <sys/ioctl.h>

int main(int argc, char *argv[])
{
    // int fd = open("/dev/tty10", O_RDONLY);
    // if (fd == -1 || argc != 3) return -1;
    // return ioctl(fd, KDMKTONE, (atoi(argv[2])<<16)+(1193180/atoi(argv[1])));
    // ioctl(STDOUT_FILENO, KIOCSOUND, 1193180 / freq);
    // usleep(wait);
    // ioctl(STDOUT_FILENO, KIOCSOUND, 0);
    int freq[] = { /* C   D    E    F    G    A    B    C */
        523, 587, 659, 698, 784, 880, 988, 1046 };
    int i;

    for (i=0; i<8; i++)
    {
        ioctl(STDOUT_FILENO, KIOCSOUND, 1193180/freq[i]);
        usleep(500000);
    }
    ioctl(STDOUT_FILENO, KIOCSOUND, 0); /*Stop silly sound*/
    return 0;
}
