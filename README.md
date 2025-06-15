How to set up:
1. Separately clone and build firecracker-microvm/firecracker
2. Copy the build contents into ./vm/firecracker
3. Use firecracker's tools to build an Ubuntu rootfs image and Linux kernel
4. Place these files under ./vm/
5. Separately clone and build sda-p/taskd/
6. Mount your Ubuntu image and place the taskd binary under /bin/
7. Add all necessary .tar files and populate the filesystem with directory targets

It's a time consuming process, but once you've done it, further adjustments are trivial.
